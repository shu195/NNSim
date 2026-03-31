import pytest

from nnsim.data import make_batches, train_validation_split


def test_train_validation_split_sizes() -> None:
    dataset = [([float(i)], [float(i % 2)]) for i in range(20)]
    split = train_validation_split(dataset, validation_split=0.2, seed=1)

    assert len(split.validation) == 4
    assert len(split.train) == 16


def test_train_validation_split_rejects_empty_train() -> None:
    dataset = [([1.0], [1.0])]
    with pytest.raises(ValueError):
        train_validation_split(dataset, validation_split=0.99, seed=1)


def test_make_batches() -> None:
    dataset = [([float(i)], [float(i)]) for i in range(10)]
    batches = make_batches(dataset, batch_size=4)

    assert [len(batch) for batch in batches] == [4, 4, 2]
