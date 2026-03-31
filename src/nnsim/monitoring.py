from __future__ import annotations

ALLOWED_MONITORS = {"train_loss", "val_loss", "val_accuracy"}


def resolve_mode(monitor: str, mode: str) -> str:
    if mode != "auto":
        return mode
    if monitor in {"train_loss", "val_loss"}:
        return "min"
    return "max"


def extract_metric(
    monitor: str,
    train_loss: float,
    val_loss: float | None,
    val_accuracy: float | None,
) -> float | None:
    if monitor == "train_loss":
        return train_loss
    if monitor == "val_loss":
        return val_loss
    if monitor == "val_accuracy":
        return val_accuracy
    raise ValueError("Unknown monitor. Use train_loss, val_loss, or val_accuracy")


def initial_best_value(mode: str) -> float:
    if mode == "min":
        return float("inf")
    return float("-inf")


def is_better(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return (best - current) > min_delta
    return (current - best) > min_delta


def is_tie(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return abs(best - current) <= min_delta
    return abs(current - best) <= min_delta
