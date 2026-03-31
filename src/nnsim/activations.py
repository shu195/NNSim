from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Activation:
    name: str
    fn: Callable[[float], float]
    grad_from_output: Callable[[float], float]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def d_sigmoid_from_output(y: float) -> float:
    return y * (1.0 - y)


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def d_relu_from_output(y: float) -> float:
    return 1.0 if y > 0.0 else 0.0


ACTIVATIONS: dict[str, Activation] = {
    "sigmoid": Activation("sigmoid", sigmoid, d_sigmoid_from_output),
    "relu": Activation("relu", relu, d_relu_from_output),
}


def get_activation(name: str) -> Activation:
    lowered = name.lower()
    if lowered not in ACTIVATIONS:
        allowed = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Unknown activation '{name}'. Allowed: {allowed}")
    return ACTIVATIONS[lowered]
