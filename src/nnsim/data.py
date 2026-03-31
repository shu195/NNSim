from __future__ import annotations

import random
from dataclasses import dataclass

from .types import Dataset


@dataclass(frozen=True)
class SplitDataset:
    train: Dataset
    validation: Dataset


def train_validation_split(dataset: Dataset, validation_split: float, seed: int) -> SplitDataset:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0)")

    data = dataset[:]
    rng = random.Random(seed)
    rng.shuffle(data)

    if validation_split == 0.0:
        return SplitDataset(train=data, validation=[])

    validation_size = max(1, int(len(data) * validation_split))
    validation = data[:validation_size]
    train = data[validation_size:]
    if not train:
        raise ValueError("validation_split too large; no training samples remain")

    return SplitDataset(train=train, validation=validation)


def make_batches(dataset: Dataset, batch_size: int) -> list[Dataset]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
