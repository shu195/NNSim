from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "xor"
    sample_count: int = 256


@dataclass(frozen=True)
class NetworkConfig:
    layer_sizes: list[int]
    hidden_activation: str = "relu"
    lattice_indices: list[int] | None = None
    reservoir_indices: list[int] | None = None
    reservoir_sparsity: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 0.5
    optimizer: str = "sgd"
    momentum: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_value: float | None = None
    epochs: int = 2000
    print_every: int = 200
    seed: int = 42
    batch_size: int = 16
    validation_split: float = 0.0
    monitor: str = "val_loss"
    monitor_mode: str = "auto"
    secondary_monitor: str | None = None
    secondary_monitor_mode: str = "auto"
    secondary_monitor_min_delta: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    restore_best_on_early_stop: bool = False
    scheduler: str = "constant"
    scheduler_gamma: float = 0.95
    scheduler_step_size: int = 50
    min_learning_rate: float = 1e-5
    log_to_console: bool = True
    run_name: str = "run"
    best_model_path: str | None = None
    last_model_path: str | None = None
    manifest_path: str | None = None
    registry_path: str | None = None
    leaderboard_path: str | None = None
    leaderboard_top_k: int = 20
    artifact_dir: str = "artifacts"
    save_history: bool = True
