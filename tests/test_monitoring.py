from nnsim.monitoring import extract_metric, initial_best_value, is_better, is_tie, resolve_mode


def test_resolve_mode_auto() -> None:
    assert resolve_mode("train_loss", "auto") == "min"
    assert resolve_mode("val_loss", "auto") == "min"
    assert resolve_mode("val_accuracy", "auto") == "max"


def test_extract_metric() -> None:
    assert extract_metric("train_loss", 0.4, 0.7, 0.2) == 0.4
    assert extract_metric("val_loss", 0.4, 0.7, 0.2) == 0.7
    assert extract_metric("val_accuracy", 0.4, 0.7, 0.2) == 0.2


def test_comparison_helpers() -> None:
    assert initial_best_value("min") == float("inf")
    assert initial_best_value("max") == float("-inf")
    assert is_better(0.3, 0.5, "min", 0.0)
    assert is_better(0.8, 0.6, "max", 0.0)
    assert is_tie(0.5, 0.5, "min", 0.0)
