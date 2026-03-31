from __future__ import annotations

from .types import Dataset, Vector


def binary_accuracy(dataset: Dataset, predictions: list[Vector], threshold: float = 0.5) -> float:
    if not dataset:
        raise ValueError("dataset cannot be empty")
    if len(dataset) != len(predictions):
        raise ValueError("dataset and predictions length mismatch")

    correct = 0
    for (_, y_true), y_pred in zip(dataset, predictions, strict=True):
        true_label = 1 if y_true[0] >= threshold else 0
        pred_label = 1 if y_pred[0] >= threshold else 0
        if true_label == pred_label:
            correct += 1

    return correct / len(dataset)


def binary_confusion_matrix(
    dataset: Dataset,
    predictions: list[Vector],
    threshold: float = 0.5,
) -> dict[str, int]:
    if not dataset:
        raise ValueError("dataset cannot be empty")
    if len(dataset) != len(predictions):
        raise ValueError("dataset and predictions length mismatch")

    tp = tn = fp = fn = 0
    for (_, y_true), y_pred in zip(dataset, predictions, strict=True):
        true_label = 1 if y_true[0] >= threshold else 0
        pred_label = 1 if y_pred[0] >= threshold else 0
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        else:
            fn += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
