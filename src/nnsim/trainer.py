from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from .callbacks import (
    BestValCheckpointCallback,
    ConsoleLoggerCallback,
    EpochLog,
    ExperimentManifestCallback,
    LastCheckpointCallback,
    RunRegistryCallback,
    TrainingCallback,
)
from .config import TrainerConfig
from .data import make_batches, train_validation_split
from .losses import mse_loss
from .metrics import binary_accuracy, binary_confusion_matrix
from .monitoring import extract_metric, initial_best_value, is_better, is_tie, resolve_mode
from .network import NeuralNetwork
from .optimizers import OptimizerConfig, make_optimizer
from .schedulers import SchedulerConfig, make_scheduler
from .types import Dataset


@dataclass(frozen=True)
class TrainingSummary:
    train_losses: list[float]
    val_losses: list[float]
    val_accuracies: list[float]
    learning_rates: list[float]
    final_loss: float
    final_val_loss: float | None
    final_val_accuracy: float | None
    final_val_confusion_matrix: dict[str, int] | None
    monitor: str
    secondary_monitor: str | None
    best_metric_value: float | None
    best_metric_epoch: int | None
    best_secondary_metric_value: float | None
    stopped_early: bool
    restored_best_model: bool
    epochs_completed: int


class Trainer:
    def __init__(self, model: NeuralNetwork, config: TrainerConfig) -> None:
        self.model = model
        self.config = config
        self.rng = random.Random(config.seed)
        scheduler_config = SchedulerConfig(
            kind=config.scheduler,
            gamma=config.scheduler_gamma,
            step_size=config.scheduler_step_size,
            min_lr=config.min_learning_rate,
        )
        self.scheduler = make_scheduler(scheduler_config)
        optimizer_config = OptimizerConfig(
            kind=config.optimizer,
            momentum=config.momentum,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon,
        )
        self.optimizer = make_optimizer(optimizer_config)

    def _evaluate(self, dataset: Dataset) -> tuple[float, float]:
        if not dataset:
            raise ValueError("Cannot evaluate an empty dataset")

        total_loss = 0.0
        predictions: list[list[float]] = []
        for x, y in dataset:
            pred = self.model.predict(x)
            predictions.append(pred)
            total_loss += mse_loss(y, pred)

        avg_loss = total_loss / len(dataset)
        acc = binary_accuracy(dataset, predictions)
        return avg_loss, acc

    def _callbacks(self) -> list[TrainingCallback]:
        callbacks: list[TrainingCallback] = []
        if self.config.log_to_console:
            callbacks.append(ConsoleLoggerCallback(print_every=self.config.print_every))
        if self.config.best_model_path:
            callbacks.append(
                BestValCheckpointCallback(
                    path=self.config.best_model_path,
                    monitor=self.config.monitor,
                    mode=self.config.monitor_mode,
                    secondary_monitor=self.config.secondary_monitor,
                    secondary_mode=self.config.secondary_monitor_mode,
                    secondary_min_delta=self.config.secondary_monitor_min_delta,
                    min_delta=self.config.early_stopping_min_delta,
                )
            )
        if self.config.last_model_path:
            callbacks.append(LastCheckpointCallback(path=self.config.last_model_path))
        if self.config.manifest_path:
            callbacks.append(
                ExperimentManifestCallback(
                    path=self.config.manifest_path,
                    run_name=self.config.run_name,
                )
            )
        if self.config.registry_path and self.config.leaderboard_path:
            callbacks.append(
                RunRegistryCallback(
                    registry_path=self.config.registry_path,
                    leaderboard_path=self.config.leaderboard_path,
                    run_name=self.config.run_name,
                    monitor=self.config.monitor,
                    mode=self.config.monitor_mode,
                    top_k=self.config.leaderboard_top_k,
                )
            )
        return callbacks

    def fit(self, dataset: Dataset) -> TrainingSummary:
        if not dataset:
            raise ValueError("Cannot train on an empty dataset")

        split = train_validation_split(dataset, self.config.validation_split, self.config.seed)
        train_data = split.train
        val_data = split.validation

        train_losses: list[float] = []
        val_losses: list[float] = []
        val_accuracies: list[float] = []
        learning_rates: list[float] = []
        callbacks = self._callbacks()

        for callback in callbacks:
            callback.on_train_start(self.model)

        monitor_mode = resolve_mode(self.config.monitor, self.config.monitor_mode)
        best_metric = initial_best_value(monitor_mode)
        secondary_mode: str | None = None
        best_secondary_metric: float | None = None
        if self.config.secondary_monitor is not None:
            secondary_mode = resolve_mode(
                self.config.secondary_monitor,
                self.config.secondary_monitor_mode,
            )
            best_secondary_metric = initial_best_value(secondary_mode)

        best_epoch: int | None = None
        stale_epochs = 0
        stopped_early = False

        for epoch in range(1, self.config.epochs + 1):
            self.rng.shuffle(train_data)
            learning_rate = self.scheduler.get_lr(self.config.learning_rate, epoch)
            learning_rates.append(learning_rate)

            total_loss = 0.0
            batches = make_batches(train_data, batch_size=self.config.batch_size)
            for batch in batches:
                batch_loss = 0.0
                for x, y in batch:
                    batch_loss += self.model.train_step(
                        x,
                        y,
                        learning_rate=learning_rate,
                        optimizer=self.optimizer,
                        weight_decay=self.config.weight_decay,
                        grad_clip_value=self.config.grad_clip_value,
                    )
                total_loss += batch_loss / len(batch)

            train_loss = total_loss / len(batches)
            train_losses.append(train_loss)

            val_loss: float | None = None
            val_acc: float | None = None
            if val_data:
                val_loss, val_acc = self._evaluate(val_data)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                metric_value = extract_metric(
                    self.config.monitor,
                    train_loss,
                    val_loss,
                    val_acc,
                )
                if metric_value is None:
                    raise ValueError("Validation metric unavailable for the chosen monitor")

                improved = is_better(
                    metric_value,
                    best_metric,
                    monitor_mode,
                    self.config.early_stopping_min_delta,
                )

                secondary_value: float | None = None
                if self.config.secondary_monitor is not None:
                    secondary_value = extract_metric(
                        self.config.secondary_monitor,
                        train_loss,
                        val_loss,
                        val_acc,
                    )

                if (
                    not improved
                    and self.config.secondary_monitor is not None
                    and secondary_mode is not None
                    and secondary_value is not None
                    and best_secondary_metric is not None
                    and is_tie(
                        metric_value,
                        best_metric,
                        monitor_mode,
                        self.config.early_stopping_min_delta,
                    )
                ):
                    improved = is_better(
                        secondary_value,
                        best_secondary_metric,
                        secondary_mode,
                        self.config.secondary_monitor_min_delta,
                    )

                if improved:
                    best_metric = metric_value
                    if secondary_value is not None:
                        best_secondary_metric = secondary_value
                    best_epoch = epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            else:
                metric_value = extract_metric(
                    self.config.monitor,
                    train_loss,
                    val_loss,
                    val_acc,
                )
                if metric_value is not None:
                    improved = is_better(
                        metric_value,
                        best_metric,
                        monitor_mode,
                        self.config.early_stopping_min_delta,
                    )
                    if improved:
                        best_metric = metric_value
                        best_epoch = epoch
                        stale_epochs = 0
                    else:
                        stale_epochs += 1

            epoch_log = EpochLog(
                epoch=epoch,
                learning_rate=learning_rate,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
            for callback in callbacks:
                callback.on_epoch_end(self.model, epoch_log)

            if self.config.early_stopping_patience > 0 and best_epoch is not None:
                if stale_epochs >= self.config.early_stopping_patience:
                    stopped_early = True
                    break

        restored_best_model = False
        if (
            stopped_early
            and self.config.restore_best_on_early_stop
            and self.config.best_model_path
            and os.path.exists(self.config.best_model_path)
        ):
            self.model = NeuralNetwork.load(self.config.best_model_path)
            restored_best_model = True

        final_val_confusion_matrix: dict[str, int] | None = None
        if val_data:
            predictions = [self.model.predict(x) for x, _ in val_data]
            final_val_confusion_matrix = binary_confusion_matrix(val_data, predictions)

        final_val_loss = val_losses[-1] if val_losses else None
        final_val_accuracy = val_accuracies[-1] if val_accuracies else None
        summary = TrainingSummary(
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            learning_rates=learning_rates,
            final_loss=train_losses[-1],
            final_val_loss=final_val_loss,
            final_val_accuracy=final_val_accuracy,
            final_val_confusion_matrix=final_val_confusion_matrix,
            monitor=self.config.monitor,
            secondary_monitor=self.config.secondary_monitor,
            best_metric_value=(best_metric if best_epoch is not None else None),
            best_metric_epoch=best_epoch,
            best_secondary_metric_value=(best_secondary_metric if best_epoch is not None else None),
            stopped_early=stopped_early,
            restored_best_model=restored_best_model,
            epochs_completed=len(train_losses),
        )
        for callback in callbacks:
            callback.on_train_end(self.model, summary)
        if self.config.save_history:
            self._save_history(summary)
        return summary

    def _save_history(self, summary: TrainingSummary) -> None:
        os.makedirs(self.config.artifact_dir, exist_ok=True)
        history_path = os.path.join(self.config.artifact_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_losses": summary.train_losses,
                    "val_losses": summary.val_losses,
                    "val_accuracies": summary.val_accuracies,
                    "learning_rates": summary.learning_rates,
                    "final_loss": summary.final_loss,
                    "final_val_loss": summary.final_val_loss,
                    "final_val_accuracy": summary.final_val_accuracy,
                    "final_val_confusion_matrix": summary.final_val_confusion_matrix,
                    "monitor": summary.monitor,
                    "secondary_monitor": summary.secondary_monitor,
                    "best_metric_value": summary.best_metric_value,
                    "best_metric_epoch": summary.best_metric_epoch,
                    "best_secondary_metric_value": summary.best_secondary_metric_value,
                    "stopped_early": summary.stopped_early,
                    "restored_best_model": summary.restored_best_model,
                    "epochs_completed": summary.epochs_completed,
                },
                f,
                indent=2,
            )
