from __future__ import annotations

from .types import Vector


def mse_loss(y_true: Vector, y_pred: Vector) -> float:
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred, strict=True)) / len(y_true)


def mse_grad(y_true: Vector, y_pred: Vector) -> Vector:
    scale = 2.0 / len(y_true)
    return [scale * (yp - yt) for yt, yp in zip(y_true, y_pred, strict=True)]
