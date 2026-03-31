import json
from pathlib import Path

from nnsim.config import NetworkConfig, TrainerConfig
from nnsim.datasets import make_dataset
from nnsim.network import NeuralNetwork
from nnsim.trainer import Trainer


def test_training_reduces_loss_on_xor() -> None:
    data = make_dataset("xor", seed=42)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=42)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.5,
            epochs=300,
            print_every=100,
            seed=42,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.train_losses[0] > summary.final_loss
    assert summary.final_loss < 0.30
    assert len(summary.learning_rates) == summary.epochs_completed


def test_training_with_validation_and_early_stopping() -> None:
    data = make_dataset("xor", seed=9)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 4, 1], hidden_activation="sigmoid", seed=9)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.5,
            epochs=30,
            batch_size=2,
            validation_split=0.5,
            early_stopping_patience=2,
            early_stopping_min_delta=1.0,
            print_every=50,
            seed=9,
            log_to_console=False,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.stopped_early
    assert summary.epochs_completed < 30
    assert summary.final_val_loss is not None
    assert summary.final_val_accuracy is not None
    assert summary.final_val_confusion_matrix is not None


def test_training_with_adam_optimizer() -> None:
    data = make_dataset("xor", seed=3)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=3)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.1,
            optimizer="adam",
            weight_decay=1e-4,
            grad_clip_value=1.0,
            epochs=80,
            batch_size=2,
            print_every=200,
            seed=3,
            log_to_console=False,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.final_loss < summary.train_losses[0]


def test_checkpoint_and_manifest_callbacks(tmp_path: Path) -> None:
    data = make_dataset("xor", seed=12)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=12)
    )
    artifact_dir = tmp_path / "artifacts"
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.2,
            optimizer="adam",
            epochs=20,
            batch_size=2,
            validation_split=0.5,
            print_every=200,
            seed=12,
            log_to_console=False,
            run_name="callback-test",
            best_model_path=str(artifact_dir / "model_best.json"),
            last_model_path=str(artifact_dir / "model_last.json"),
            manifest_path=str(artifact_dir / "experiment_manifest.json"),
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.final_val_confusion_matrix is not None

    best_model = artifact_dir / "model_best.json"
    last_model = artifact_dir / "model_last.json"
    manifest = artifact_dir / "experiment_manifest.json"

    assert best_model.exists()
    assert last_model.exists()
    assert manifest.exists()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["run_name"] == "callback-test"
    assert "summary" in payload


def test_monitor_accuracy_mode_tracks_best_epoch() -> None:
    data = make_dataset("xor", seed=21)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=21)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.2,
            optimizer="adam",
            epochs=25,
            batch_size=2,
            validation_split=0.5,
            monitor="val_accuracy",
            monitor_mode="max",
            print_every=200,
            seed=21,
            log_to_console=False,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.monitor == "val_accuracy"
    assert summary.best_metric_epoch is not None
    assert summary.best_metric_value is not None


def test_restore_best_model_on_early_stop(tmp_path: Path) -> None:
    data = make_dataset("xor", seed=13)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=13)
    )
    best_path = tmp_path / "model_best.json"
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.2,
            optimizer="adam",
            epochs=40,
            batch_size=2,
            validation_split=0.5,
            monitor="val_loss",
            monitor_mode="min",
            early_stopping_patience=2,
            early_stopping_min_delta=1.0,
            restore_best_on_early_stop=True,
            best_model_path=str(best_path),
            print_every=200,
            seed=13,
            log_to_console=False,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.stopped_early
    assert summary.restored_best_model
    assert best_path.exists()


def test_train_loss_monitor_without_validation() -> None:
    data = make_dataset("xor", seed=31)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=31)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.2,
            optimizer="adam",
            epochs=20,
            batch_size=2,
            validation_split=0.0,
            monitor="train_loss",
            monitor_mode="min",
            early_stopping_patience=3,
            early_stopping_min_delta=0.0,
            print_every=200,
            seed=31,
            log_to_console=False,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.best_metric_epoch is not None
    assert summary.best_metric_value is not None
    assert summary.final_val_loss is None


def test_secondary_monitor_tie_break_configuration() -> None:
    data = make_dataset("xor", seed=41)
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=41)
    )
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=0.2,
            optimizer="adam",
            epochs=20,
            batch_size=2,
            validation_split=0.5,
            monitor="val_accuracy",
            monitor_mode="max",
            secondary_monitor="val_loss",
            secondary_monitor_mode="min",
            secondary_monitor_min_delta=0.0,
            print_every=200,
            seed=41,
            log_to_console=False,
            save_history=False,
        ),
    )

    summary = trainer.fit(data)
    assert summary.secondary_monitor == "val_loss"
    assert summary.best_metric_epoch is not None


def test_run_registry_and_leaderboard(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "registry_artifacts"

    for seed in [51, 52]:
        data = make_dataset("xor", seed=seed)
        model = NeuralNetwork.from_config(
            NetworkConfig(layer_sizes=[2, 6, 1], hidden_activation="sigmoid", seed=seed)
        )
        trainer = Trainer(
            model,
            TrainerConfig(
                learning_rate=0.2,
                optimizer="adam",
                epochs=8,
                batch_size=2,
                validation_split=0.5,
                monitor="val_loss",
                monitor_mode="min",
                print_every=200,
                seed=seed,
                log_to_console=False,
                run_name=f"run-{seed}",
                registry_path=str(artifact_dir / "run_registry.jsonl"),
                leaderboard_path=str(artifact_dir / "leaderboard.json"),
                leaderboard_top_k=5,
                save_history=False,
            ),
        )
        trainer.fit(data)

    registry_path = artifact_dir / "run_registry.jsonl"
    leaderboard_path = artifact_dir / "leaderboard.json"
    assert registry_path.exists()
    assert leaderboard_path.exists()

    lines = [line for line in registry_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) >= 2

    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    assert leaderboard["monitor"] == "val_loss"
    assert leaderboard["monitor_mode"] == "min"
    assert len(leaderboard["entries"]) >= 1
