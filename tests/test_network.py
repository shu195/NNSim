import pytest

from nnsim.config import NetworkConfig
from nnsim.network import NeuralNetwork


def test_predict_output_size() -> None:
    config = NetworkConfig(layer_sizes=[2, 4, 1], hidden_activation="sigmoid", seed=1)
    model = NeuralNetwork.from_config(config)
    pred = model.predict([1.0, 0.0])
    assert len(pred) == 1


def test_overlap_indices_fail() -> None:
    config = NetworkConfig(
        layer_sizes=[2, 4, 1],
        hidden_activation="relu",
        lattice_indices=[0],
        reservoir_indices=[0],
        seed=1,
    )
    with pytest.raises(ValueError) as exc_info:
        NeuralNetwork.from_config(config)

    assert "both lattice and reservoir" in str(exc_info.value)
