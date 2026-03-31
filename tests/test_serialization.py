from pathlib import Path

from nnsim.config import NetworkConfig
from nnsim.network import NeuralNetwork


def test_model_save_load_roundtrip(tmp_path: Path) -> None:
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 5, 1], hidden_activation="sigmoid", seed=7)
    )

    x = [0.2, 0.8]
    before = model.predict(x)

    model_path = tmp_path / "model.json"
    model.save(str(model_path))

    loaded = NeuralNetwork.load(str(model_path))
    after = loaded.predict(x)

    assert len(before) == len(after)
    for b, a in zip(before, after, strict=True):
        assert abs(b - a) < 1e-12


def test_loaded_model_keeps_training(tmp_path: Path) -> None:
    model = NeuralNetwork.from_config(
        NetworkConfig(layer_sizes=[2, 4, 1], hidden_activation="sigmoid", seed=11)
    )
    model_path = tmp_path / "resume.json"
    model.save(str(model_path))

    loaded = NeuralNetwork.load(str(model_path))
    loss = loaded.train_step([1.0, 0.0], [1.0], learning_rate=0.3)

    assert loss >= 0.0
