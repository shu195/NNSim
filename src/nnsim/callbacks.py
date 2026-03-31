from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .monitoring import extract_metric, initial_best_value, is_better, is_tie, resolve_mode

if TYPE_CHECKING:
    from .network import NeuralNetwork
    from .trainer import TrainingSummary


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    learning_rate: float
    train_loss: float
    val_loss: float | None
    val_accuracy: float | None


class TrainingCallback:
    def on_train_start(self, model: NeuralNetwork) -> None:
        del model
        return

    def on_epoch_end(self, model: NeuralNetwork, log: EpochLog) -> None:
        del model, log
        return

    def on_train_end(self, model: NeuralNetwork, summary: TrainingSummary) -> None:
        del model, summary
        return


class ConsoleLoggerCallback(TrainingCallback):
    def __init__(self, print_every: int) -> None:
        self.print_every = max(1, print_every)

    def on_epoch_end(self, model: NeuralNetwork, log: EpochLog) -> None:
        del model
        if log.epoch == 1 or log.epoch % self.print_every == 0:
            if log.val_loss is None:
                print(
                    f"Epoch {log.epoch:5d} | lr: {log.learning_rate:.6f} "
                    f"| train loss: {log.train_loss:.6f}"
                )
            else:
                print(
                    "Epoch "
                    f"{log.epoch:5d} | lr: {log.learning_rate:.6f} "
                    f"| train loss: {log.train_loss:.6f} "
                    f"| val loss: {log.val_loss:.6f} | val acc: {log.val_accuracy:.4f}"
                )


class BestValCheckpointCallback(TrainingCallback):
    def __init__(
        self,
        path: str,
        monitor: str = "val_loss",
        mode: str = "auto",
        secondary_monitor: str | None = None,
        secondary_mode: str = "auto",
        secondary_min_delta: float = 0.0,
        min_delta: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.secondary_monitor = secondary_monitor
        self.secondary_mode = secondary_mode
        self.secondary_min_delta = secondary_min_delta
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.best_secondary_value: float | None = None
        self._is_min_mode = True
        self._primary_mode = "min"
        self._secondary_mode: str | None = None

    def on_train_start(self, model: NeuralNetwork) -> None:
        del model
        self._primary_mode = resolve_mode(self.monitor, self.mode)
        self._is_min_mode = self._primary_mode == "min"
        self.best_value = initial_best_value(self._primary_mode)

        if self.secondary_monitor is not None:
            self._secondary_mode = resolve_mode(self.secondary_monitor, self.secondary_mode)
            self.best_secondary_value = initial_best_value(self._secondary_mode)
        else:
            self._secondary_mode = None
            self.best_secondary_value = None

    def on_epoch_end(self, model: NeuralNetwork, log: EpochLog) -> None:
        primary = extract_metric(self.monitor, log.train_loss, log.val_loss, log.val_accuracy)
        if primary is None:
            return

        improved = is_better(primary, self.best_value, self._primary_mode, self.min_delta)

        secondary = None
        if self.secondary_monitor is not None:
            secondary = extract_metric(
                self.secondary_monitor,
                log.train_loss,
                log.val_loss,
                log.val_accuracy,
            )

        if (
            not improved
            and self.secondary_monitor is not None
            and self._secondary_mode is not None
            and secondary is not None
            and self.best_secondary_value is not None
            and is_tie(primary, self.best_value, self._primary_mode, self.min_delta)
        ):
            improved = is_better(
                secondary,
                self.best_secondary_value,
                self._secondary_mode,
                self.secondary_min_delta,
            )

        if improved:
            self.best_value = primary
            if secondary is not None:
                self.best_secondary_value = secondary
            self.path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.path))


class LastCheckpointCallback(TrainingCallback):
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def on_train_end(self, model: NeuralNetwork, summary: TrainingSummary) -> None:
        del summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(self.path))


class ExperimentManifestCallback(TrainingCallback):
    def __init__(self, path: str, run_name: str) -> None:
        self.path = Path(path)
        self.run_name = run_name

    def on_train_end(self, model: NeuralNetwork, summary: TrainingSummary) -> None:
        payload = {
            "run_name": self.run_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "network_config": {
                "layer_sizes": model.config.layer_sizes,
                "hidden_activation": model.config.hidden_activation,
                "lattice_indices": model.config.lattice_indices,
                "reservoir_indices": model.config.reservoir_indices,
                "reservoir_sparsity": model.config.reservoir_sparsity,
                "seed": model.config.seed,
            },
            "summary": asdict(summary),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


class RunRegistryCallback(TrainingCallback):
    def __init__(
        self,
        registry_path: str,
        leaderboard_path: str,
        run_name: str,
        monitor: str,
        mode: str,
        top_k: int,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.leaderboard_path = Path(leaderboard_path)
        self.run_name = run_name
        self.monitor = monitor
        self.mode = resolve_mode(monitor, mode)
        self.top_k = max(1, top_k)

    def _load_registry(self) -> list[dict[str, object]]:
        if not self.registry_path.exists():
            return []

        entries: list[dict[str, object]] = []
        with self.registry_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    entries.append(json.loads(text))
        return entries

    def _score(self, item: dict[str, object]) -> float:
        score_obj = item.get("monitor_value")
        if isinstance(score_obj, (int, float)):
            return float(score_obj)
        return float("inf") if self.mode == "min" else float("-inf")

    def on_train_end(self, model: NeuralNetwork, summary: TrainingSummary) -> None:
        del model

        monitor_value: float | None
        if self.monitor == "train_loss":
            monitor_value = summary.final_loss
        elif self.monitor == "val_loss":
            monitor_value = summary.final_val_loss
        elif self.monitor == "val_accuracy":
            monitor_value = summary.final_val_accuracy
        else:
            raise ValueError("Unknown monitor. Use train_loss, val_loss, or val_accuracy")

        entry: dict[str, object] = {
            "run_name": self.run_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "monitor": self.monitor,
            "monitor_mode": self.mode,
            "monitor_value": monitor_value,
            "stopped_early": summary.stopped_early,
            "epochs_completed": summary.epochs_completed,
            "final_loss": summary.final_loss,
            "final_val_loss": summary.final_val_loss,
            "final_val_accuracy": summary.final_val_accuracy,
        }

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        records = self._load_registry()
        filtered = [
            row
            for row in records
            if isinstance(row.get("monitor"), str)
            and isinstance(row.get("monitor_mode"), str)
            and row["monitor"] == self.monitor
            and row["monitor_mode"] == self.mode
        ]

        ranked = sorted(
            filtered,
            key=self._score,
            reverse=(self.mode == "max"),
        )[: self.top_k]

        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        with self.leaderboard_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "monitor": self.monitor,
                    "monitor_mode": self.mode,
                    "top_k": self.top_k,
                    "entries": ranked,
                },
                fh,
                indent=2,
            )
