from __future__ import annotations

import argparse

from .config import DatasetConfig, NetworkConfig, TrainerConfig
from .datasets import infer_io_shape, make_dataset
from .network import NeuralNetwork
from .trainer import Trainer
from .types import Vector


def _parse_layers(text: str) -> list[int]:
    layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(layers) < 2:
        raise ValueError("Layers must include at least input and output size")
    return layers


def _parse_indices(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()] if text else []


def _format_vector(values: Vector) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _print_predictions(model: NeuralNetwork, dataset: list[tuple[Vector, Vector]]) -> None:
    print("\nSample predictions:")
    for x, y in dataset:
        pred = model.predict(x)
        print(f"  x={_format_vector(x)}  y={_format_vector(y)}  pred={_format_vector(pred)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional neural network simulator")
    parser.add_argument(
        "--dataset",
        type=str,
        default="xor",
        choices=["xor", "and", "or", "circles", "spiral", "text"],
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--layers", type=str, default="2,4,1")
    parser.add_argument("--lattice", type=str, default="")
    parser.add_argument("--reservoir", type=str, default="")
    parser.add_argument("--reservoir-sparsity", type=float, default=0.15)
    parser.add_argument("--activation", type=str, default="sigmoid", choices=["relu", "sigmoid"])
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=-1.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["train_loss", "val_loss", "val_accuracy"],
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        default="auto",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--secondary-monitor",
        type=str,
        default="",
        choices=["", "train_loss", "val_loss", "val_accuracy"],
    )
    parser.add_argument(
        "--secondary-monitor-mode",
        type=str,
        default="auto",
        choices=["auto", "min", "max"],
    )
    parser.add_argument("--secondary-monitor-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--restore-best-on-early-stop", action="store_true")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        choices=["constant", "exp", "step"],
    )
    parser.add_argument("--scheduler-gamma", type=float, default=0.95)
    parser.add_argument("--scheduler-step-size", type=int, default=50)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--no-save-history", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--save-best-model", action="store_true")
    parser.add_argument("--save-last-model", action="store_true")
    parser.add_argument("--save-manifest", action="store_true")
    parser.add_argument("--save-registry", action="store_true")
    parser.add_argument("--leaderboard-top-k", type=int, default=20)
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    return parser


def run(args: argparse.Namespace) -> int:
    dataset_cfg = DatasetConfig(name=args.dataset, sample_count=args.samples)
    data = make_dataset(dataset_cfg.name, sample_count=dataset_cfg.sample_count, seed=args.seed)

    if args.load_model:
        model = NeuralNetwork.load(args.load_model)
        layers = model.config.layer_sizes
    else:
        layers = _parse_layers(args.layers)
        network_cfg = NetworkConfig(
            layer_sizes=layers,
            hidden_activation=args.activation,
            lattice_indices=_parse_indices(args.lattice),
            reservoir_indices=_parse_indices(args.reservoir),
            reservoir_sparsity=args.reservoir_sparsity,
            seed=args.seed,
        )
        model = NeuralNetwork.from_config(network_cfg)

    in_size, out_size = infer_io_shape(data)
    if layers[0] != in_size:
        raise ValueError(f"Input size mismatch. Dataset expects {in_size}, got {layers[0]}")
    if layers[-1] != out_size:
        raise ValueError(f"Output size mismatch. Dataset expects {out_size}, got {layers[-1]}")

    trainer_cfg = TrainerConfig(
        learning_rate=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        grad_clip_value=(None if args.grad_clip < 0.0 else args.grad_clip),
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        secondary_monitor=(args.secondary_monitor if args.secondary_monitor else None),
        secondary_monitor_mode=args.secondary_monitor_mode,
        secondary_monitor_min_delta=args.secondary_monitor_min_delta,
        early_stopping_patience=args.early_stop_patience,
        early_stopping_min_delta=args.early_stop_min_delta,
        restore_best_on_early_stop=args.restore_best_on_early_stop,
        scheduler=args.scheduler,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        min_learning_rate=args.min_lr,
        print_every=args.print_every,
        seed=args.seed,
        log_to_console=not args.quiet,
        run_name=args.run_name,
        best_model_path=(
            f"{args.artifact_dir}/model_best.json" if args.save_best_model else None
        ),
        last_model_path=(
            f"{args.artifact_dir}/model_last.json" if args.save_last_model else None
        ),
        manifest_path=(
            f"{args.artifact_dir}/experiment_manifest.json" if args.save_manifest else None
        ),
        registry_path=(
            f"{args.artifact_dir}/run_registry.jsonl" if args.save_registry else None
        ),
        leaderboard_path=(
            f"{args.artifact_dir}/leaderboard.json" if args.save_registry else None
        ),
        leaderboard_top_k=args.leaderboard_top_k,
        artifact_dir=args.artifact_dir,
        save_history=not args.no_save_history,
    )
    trainer = Trainer(model, trainer_cfg)

    print("Neural Network Simulator")
    print("=" * 24)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Layers: {layers}")
    print(f"Activation (hidden): {args.activation}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validation split: {args.val_split}")
    print(f"Monitor: {args.monitor} ({args.monitor_mode})")
    if args.secondary_monitor:
        print(
            "Secondary monitor: "
            f"{args.secondary_monitor} ({args.secondary_monitor_mode})"
        )

    summary = trainer.fit(data)
    if args.save_model:
        model.save(args.save_model)
        print(f"Saved model: {args.save_model}")
    if summary.stopped_early:
        print(f"Early stopping activated at epoch {summary.epochs_completed}")
    if summary.best_metric_epoch is not None and summary.best_metric_value is not None:
        print(
            f"Best {summary.monitor}: {summary.best_metric_value:.6f} "
            f"at epoch {summary.best_metric_epoch}"
        )
    if (
        summary.secondary_monitor is not None
        and summary.best_secondary_metric_value is not None
    ):
        print(
            f"Best {summary.secondary_monitor}: "
            f"{summary.best_secondary_metric_value:.6f}"
        )
    if summary.restored_best_model:
        print("Restored best model checkpoint after early stopping")
    print(f"Final loss: {summary.final_loss:.6f}")
    if summary.final_val_loss is not None and summary.final_val_accuracy is not None:
        print(f"Final val loss: {summary.final_val_loss:.6f}")
        print(f"Final val accuracy: {summary.final_val_accuracy:.4f}")
    if summary.final_val_confusion_matrix is not None:
        matrix = summary.final_val_confusion_matrix
        print(
            "Final val confusion matrix: "
            f"TP={matrix['tp']} TN={matrix['tn']} FP={matrix['fp']} FN={matrix['fn']}"
        )
    _print_predictions(model, data)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)
