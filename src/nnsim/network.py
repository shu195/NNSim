from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .activations import get_activation
from .config import NetworkConfig
from .layers import DenseLayer, LatticeLayer, Layer, ReservoirLayer
from .losses import mse_grad, mse_loss
from .optimizers import Optimizer
from .types import Vector


@dataclass
class NeuralNetwork:
    config: NetworkConfig
    layers: list[Layer]

    @classmethod
    def from_config(cls, config: NetworkConfig) -> NeuralNetwork:
        if len(config.layer_sizes) < 2:
            raise ValueError("Need at least two layer sizes: input and output")

        lattice_indices = set(config.lattice_indices or [])
        reservoir_indices = set(config.reservoir_indices or [])
        overlap = lattice_indices & reservoir_indices
        if overlap:
            raise ValueError(f"Indices cannot be both lattice and reservoir: {sorted(overlap)}")

        output_layer_idx = len(config.layer_sizes) - 2
        if output_layer_idx in reservoir_indices:
            raise ValueError(
                "Final layer cannot be reservoir because output layer must be trainable"
            )

        rng = random.Random(config.seed)
        layers: list[Layer] = []

        for idx in range(len(config.layer_sizes) - 1):
            in_size = config.layer_sizes[idx]
            out_size = config.layer_sizes[idx + 1]
            if idx in reservoir_indices:
                layers.append(
                    ReservoirLayer.build(
                        in_size=in_size,
                        out_size=out_size,
                        activation=config.hidden_activation,
                        sparsity=config.reservoir_sparsity,
                        rng=rng,
                    )
                )
                continue

            if idx in lattice_indices:
                in_sizes: list[int] = []
                if idx > 0:
                    in_sizes.append(config.layer_sizes[idx])
                if idx > 1:
                    in_sizes.append(config.layer_sizes[idx - 1])
                if not in_sizes:
                    in_sizes = [in_size]
                layers.append(
                    LatticeLayer.build(
                        in_sizes=in_sizes,
                        out_size=out_size,
                        activation=config.hidden_activation,
                        rng=rng,
                    )
                )
                continue

            activation = (
                config.hidden_activation if idx < len(config.layer_sizes) - 2 else "sigmoid"
            )
            layers.append(DenseLayer.build(in_size, out_size, activation, rng))

        return cls(config=config, layers=layers)

    def forward(self, x: Vector) -> Vector:
        states = [x]
        for layer in self.layers:
            if isinstance(layer, LatticeLayer):
                needed = len(layer.in_sizes)
                inputs = states[-needed:] if len(states) >= needed else [states[-1]] * needed
                out = layer.forward(inputs)
            else:
                out = layer.forward(states[-1])
            states.append(out)
        return states[-1]

    def train_step(
        self,
        x: Vector,
        y_true: Vector,
        learning_rate: float,
        optimizer: Optimizer | None = None,
        weight_decay: float = 0.0,
        grad_clip_value: float | None = None,
    ) -> float:
        y_pred = self.forward(x)
        loss = mse_loss(y_true, y_pred)
        grad = mse_grad(y_true, y_pred)

        for layer_idx, layer in reversed(list(enumerate(self.layers))):
            layer_key = f"layer{layer_idx}"
            if isinstance(layer, LatticeLayer):
                grad_list = layer.backward(
                    grad_output=grad,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    layer_key=layer_key,
                    weight_decay=weight_decay,
                    grad_clip_value=grad_clip_value,
                )
                grad = [sum(values) for values in zip(*grad_list, strict=True)]
            else:
                grad = layer.backward(
                    grad_output=grad,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    layer_key=layer_key,
                    weight_decay=weight_decay,
                    grad_clip_value=grad_clip_value,
                )

        return loss

    def predict(self, x: Vector) -> Vector:
        return self.forward(x)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": {
                "layer_sizes": self.config.layer_sizes,
                "hidden_activation": self.config.hidden_activation,
                "lattice_indices": self.config.lattice_indices,
                "reservoir_indices": self.config.reservoir_indices,
                "reservoir_sparsity": self.config.reservoir_sparsity,
                "seed": self.config.seed,
            },
            "layers": [],
        }

        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                payload["layers"].append(
                    {
                        "type": "dense",
                        "in_size": layer.in_size,
                        "out_size": layer.out_size,
                        "activation": layer.activation.name,
                        "weights": layer.weights,
                        "biases": layer.biases,
                    }
                )
                continue

            if isinstance(layer, ReservoirLayer):
                payload["layers"].append(
                    {
                        "type": "reservoir",
                        "in_size": layer.in_size,
                        "out_size": layer.out_size,
                        "activation": layer.activation.name,
                        "weights": layer.weights,
                        "biases": layer.biases,
                    }
                )
                continue

            payload["layers"].append(
                {
                    "type": "lattice",
                    "in_sizes": layer.in_sizes,
                    "out_size": layer.out_size,
                    "activation": layer.activation.name,
                    "weights": layer.weights,
                    "biases": layer.biases,
                }
            )

        return payload

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> NeuralNetwork:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        config = NetworkConfig(**payload["config"])
        layers: list[Layer] = []

        for data in payload["layers"]:
            layer_type = data["type"]
            if layer_type == "dense":
                layers.append(
                    DenseLayer(
                        in_size=data["in_size"],
                        out_size=data["out_size"],
                        activation=get_activation(data["activation"]),
                        weights=data["weights"],
                        biases=data["biases"],
                        last_input=[],
                        last_output=[],
                    )
                )
                continue

            if layer_type == "reservoir":
                layers.append(
                    ReservoirLayer(
                        in_size=data["in_size"],
                        out_size=data["out_size"],
                        activation=get_activation(data["activation"]),
                        weights=data["weights"],
                        biases=data["biases"],
                        last_input=[],
                        last_output=[],
                    )
                )
                continue

            if layer_type == "lattice":
                layers.append(
                    LatticeLayer(
                        in_sizes=data["in_sizes"],
                        out_size=data["out_size"],
                        activation=get_activation(data["activation"]),
                        weights=data["weights"],
                        biases=data["biases"],
                        last_inputs=[],
                        last_output=[],
                    )
                )
                continue

            raise ValueError(f"Unknown layer type in model payload: {layer_type}")

        return cls(config=config, layers=layers)
