from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .activations import Activation, get_activation
from .types import Matrix, Vector

if TYPE_CHECKING:
    from .optimizers import Optimizer


def _clamp(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    return max(-limit, min(limit, value))


@dataclass
class DenseLayer:
    in_size: int
    out_size: int
    activation: Activation
    weights: Matrix
    biases: Vector
    last_input: Vector
    last_output: Vector

    @classmethod
    def build(
        cls,
        in_size: int,
        out_size: int,
        activation: str,
        rng: random.Random,
    ) -> DenseLayer:
        scale = math.sqrt(2.0 / max(1, in_size))
        weights = [
            [rng.uniform(-1.0, 1.0) * scale for _ in range(in_size)]
            for _ in range(out_size)
        ]
        return cls(
            in_size=in_size,
            out_size=out_size,
            activation=get_activation(activation),
            weights=weights,
            biases=[0.0 for _ in range(out_size)],
            last_input=[],
            last_output=[],
        )

    def forward(self, x: Vector) -> Vector:
        self.last_input = x[:]
        out = [
            self.activation.fn(sum(w * xi for w, xi in zip(ws, x, strict=True)) + b)
            for ws, b in zip(self.weights, self.biases, strict=True)
        ]
        self.last_output = out
        return out

    def backward(
        self,
        grad_output: Vector,
        learning_rate: float,
        optimizer: Optimizer | None = None,
        layer_key: str = "",
        weight_decay: float = 0.0,
        grad_clip_value: float | None = None,
    ) -> Vector:
        grad_z = [
            go * self.activation.grad_from_output(y)
            for go, y in zip(grad_output, self.last_output, strict=True)
        ]
        grad_input = [0.0 for _ in range(self.in_size)]

        for j in range(self.out_size):
            for i in range(self.in_size):
                grad_input[i] += grad_z[j] * self.weights[j][i]
                grad_w = grad_z[j] * self.last_input[i] + weight_decay * self.weights[j][i]
                grad_w = _clamp(grad_w, grad_clip_value)
                if optimizer is None:
                    self.weights[j][i] -= learning_rate * grad_w
                else:
                    key = f"{layer_key}.weights.{j}.{i}"
                    self.weights[j][i] = optimizer.update(
                        key=key,
                        value=self.weights[j][i],
                        grad=grad_w,
                        learning_rate=learning_rate,
                    )

            grad_b = _clamp(grad_z[j], grad_clip_value)
            if optimizer is None:
                self.biases[j] -= learning_rate * grad_b
            else:
                bias_key = f"{layer_key}.biases.{j}"
                self.biases[j] = optimizer.update(
                    key=bias_key,
                    value=self.biases[j],
                    grad=grad_b,
                    learning_rate=learning_rate,
                )

        return grad_input


@dataclass
class ReservoirLayer:
    in_size: int
    out_size: int
    activation: Activation
    weights: Matrix
    biases: Vector
    last_input: Vector
    last_output: Vector

    @classmethod
    def build(
        cls,
        in_size: int,
        out_size: int,
        activation: str,
        sparsity: float,
        rng: random.Random,
    ) -> ReservoirLayer:
        scale = math.sqrt(2.0 / max(1, in_size))
        weights: Matrix = []
        for _ in range(out_size):
            row: Vector = []
            for _ in range(in_size):
                if rng.random() < sparsity:
                    row.append(rng.uniform(-1.0, 1.0) * scale)
                else:
                    row.append(0.0)
            weights.append(row)
        return cls(
            in_size=in_size,
            out_size=out_size,
            activation=get_activation(activation),
            weights=weights,
            biases=[0.0 for _ in range(out_size)],
            last_input=[],
            last_output=[],
        )

    def forward(self, x: Vector) -> Vector:
        self.last_input = x[:]
        out = [
            self.activation.fn(sum(w * xi for w, xi in zip(ws, x, strict=True)) + b)
            for ws, b in zip(self.weights, self.biases, strict=True)
        ]
        self.last_output = out
        return out

    def backward(
        self,
        grad_output: Vector,
        learning_rate: float,
        optimizer: Optimizer | None = None,
        layer_key: str = "",
        weight_decay: float = 0.0,
        grad_clip_value: float | None = None,
    ) -> Vector:
        del learning_rate, optimizer, layer_key, weight_decay, grad_clip_value
        grad_z = [
            go * self.activation.grad_from_output(y)
            for go, y in zip(grad_output, self.last_output, strict=True)
        ]
        grad_input = [0.0 for _ in range(self.in_size)]
        for j in range(self.out_size):
            for i in range(self.in_size):
                grad_input[i] += grad_z[j] * self.weights[j][i]
        return grad_input


@dataclass
class LatticeLayer:
    in_sizes: list[int]
    out_size: int
    activation: Activation
    weights: list[Matrix]
    biases: Vector
    last_inputs: list[Vector]
    last_output: Vector

    @classmethod
    def build(
        cls,
        in_sizes: list[int],
        out_size: int,
        activation: str,
        rng: random.Random,
    ) -> LatticeLayer:
        weight_sets: list[Matrix] = []
        for size in in_sizes:
            scale = math.sqrt(2.0 / max(1, size))
            weight_sets.append([
                [rng.uniform(-1.0, 1.0) * scale for _ in range(size)]
                for _ in range(out_size)
            ])
        return cls(
            in_sizes=in_sizes,
            out_size=out_size,
            activation=get_activation(activation),
            weights=weight_sets,
            biases=[0.0 for _ in range(out_size)],
            last_inputs=[],
            last_output=[],
        )

    def forward(self, inputs: list[Vector]) -> Vector:
        self.last_inputs = [v[:] for v in inputs]
        z = [0.0 for _ in range(self.out_size)]
        for input_vec, weight_block in zip(inputs, self.weights, strict=True):
            for j in range(self.out_size):
                z[j] += sum(
                    w * x for w, x in zip(weight_block[j], input_vec, strict=True)
                )
        out = [self.activation.fn(zj + bj) for zj, bj in zip(z, self.biases, strict=True)]
        self.last_output = out
        return out

    def backward(
        self,
        grad_output: Vector,
        learning_rate: float,
        optimizer: Optimizer | None = None,
        layer_key: str = "",
        weight_decay: float = 0.0,
        grad_clip_value: float | None = None,
    ) -> list[Vector]:
        grad_z = [
            go * self.activation.grad_from_output(y)
            for go, y in zip(grad_output, self.last_output, strict=True)
        ]
        grad_inputs = [[0.0 for _ in range(size)] for size in self.in_sizes]

        for j in range(self.out_size):
            for b_idx, block in enumerate(self.weights):
                for i in range(self.in_sizes[b_idx]):
                    grad_inputs[b_idx][i] += grad_z[j] * block[j][i]
                    grad_w = grad_z[j] * self.last_inputs[b_idx][i] + weight_decay * block[j][i]
                    grad_w = _clamp(grad_w, grad_clip_value)
                    if optimizer is None:
                        block[j][i] -= learning_rate * grad_w
                    else:
                        key = f"{layer_key}.weights.{b_idx}.{j}.{i}"
                        block[j][i] = optimizer.update(
                            key=key,
                            value=block[j][i],
                            grad=grad_w,
                            learning_rate=learning_rate,
                        )

            grad_b = _clamp(grad_z[j], grad_clip_value)
            if optimizer is None:
                self.biases[j] -= learning_rate * grad_b
            else:
                bias_key = f"{layer_key}.biases.{j}"
                self.biases[j] = optimizer.update(
                    key=bias_key,
                    value=self.biases[j],
                    grad=grad_b,
                    learning_rate=learning_rate,
                )

        return grad_inputs


Layer = DenseLayer | ReservoirLayer | LatticeLayer
