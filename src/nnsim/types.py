from typing import TypeAlias

Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[list[float]]
Sample: TypeAlias = tuple[Vector, Vector]
Dataset: TypeAlias = list[Sample]
