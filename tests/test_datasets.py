from nnsim.datasets import infer_io_shape, make_dataset


def test_xor_shape() -> None:
    data = make_dataset("xor")
    assert len(data) == 4
    assert infer_io_shape(data) == (2, 1)


def test_spiral_shape_with_samples() -> None:
    data = make_dataset("spiral", sample_count=128, seed=7)
    assert len(data) == 128
    assert infer_io_shape(data) == (2, 2)


def test_text_dataset_has_binary_labels() -> None:
    data = make_dataset("text")
    labels = {y[0] for _, y in data}
    assert labels == {0.0, 1.0}
