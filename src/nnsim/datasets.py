from __future__ import annotations

import math
import random

from .types import Dataset, Sample


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _logic_dataset(name: str) -> Dataset:
    if name == "xor":
        return [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]
    if name == "and":
        return [([0.0, 0.0], [0.0]), ([0.0, 1.0], [0.0]), ([1.0, 0.0], [0.0]), ([1.0, 1.0], [1.0])]
    if name == "or":
        return [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [1.0])]
    raise ValueError(f"Unsupported logic dataset '{name}'")


def _manifold_dataset(name: str, sample_count: int, rng: random.Random) -> Dataset:
    samples: Dataset = []
    center_x = 0.5
    center_y = 0.5

    for index in range(sample_count):
        progress = index / sample_count
        angle = progress * 4.0 * math.pi

        if name == "circles":
            base_radius = 0.16 if index % 2 == 0 else 0.32
            radius = base_radius + rng.uniform(-0.025, 0.025)
        else:
            radius = 0.06 + 0.36 * progress + rng.uniform(-0.02, 0.02)

        x = _clamp(center_x + radius * math.cos(angle), 0.0, 1.0)
        y = _clamp(center_y + radius * math.sin(angle), 0.0, 1.0)
        point = [x, y]
        samples.append((point, point[:]))

    return samples


def _text_dataset() -> Dataset:
    sentences: list[tuple[str, int]] = [
        ("I love this", 1),
        ("This is great", 1),
        ("Amazing work", 1),
        ("I hate it", 0),
        ("This sucks", 0),
        ("Terrible idea", 0),
        ("Not bad", 1),
        ("Pretty good", 1),
        ("Awful experience", 0),
        ("Fantastic", 1),
    ]
    vocab = sorted({word for sentence, _ in sentences for word in sentence.lower().split()})
    dataset: Dataset = []
    for sentence, label in sentences:
        words = set(sentence.lower().split())
        vec = [1.0 if token in words else 0.0 for token in vocab]
        dataset.append((vec, [float(label)]))
    return dataset


def make_dataset(name: str, sample_count: int = 256, seed: int = 42) -> Dataset:
    lowered = name.lower()
    if lowered in {"xor", "and", "or"}:
        return _logic_dataset(lowered)

    rng = random.Random(seed)
    if lowered in {"circles", "spiral"}:
        return _manifold_dataset(lowered, sample_count=sample_count, rng=rng)
    if lowered == "text":
        return _text_dataset()

    raise ValueError("Unknown dataset. Use xor, and, or, circles, spiral, text")


def infer_io_shape(dataset: list[Sample]) -> tuple[int, int]:
    if not dataset:
        raise ValueError("Dataset cannot be empty")
    return len(dataset[0][0]), len(dataset[0][1])
