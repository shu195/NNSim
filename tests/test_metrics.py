from nnsim.metrics import binary_confusion_matrix


def test_binary_confusion_matrix() -> None:
    dataset = [([0.0], [0.0]), ([1.0], [1.0]), ([2.0], [0.0]), ([3.0], [1.0])]
    predictions = [[0.1], [0.9], [0.7], [0.3]]

    matrix = binary_confusion_matrix(dataset, predictions)
    assert matrix == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
