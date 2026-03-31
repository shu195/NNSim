from nnsim.optimizers import OptimizerConfig, make_optimizer


def test_sgd_optimizer_update() -> None:
    optimizer = make_optimizer(OptimizerConfig(kind="sgd"))
    updated = optimizer.update("p", value=1.0, grad=0.5, learning_rate=0.1)
    assert abs(updated - 0.95) < 1e-12


def test_momentum_optimizer_accumulates_velocity() -> None:
    optimizer = make_optimizer(OptimizerConfig(kind="momentum", momentum=0.9))
    first = optimizer.update("p", value=1.0, grad=1.0, learning_rate=0.1)
    second = optimizer.update("p", value=first, grad=1.0, learning_rate=0.1)

    assert second < first


def test_adam_optimizer_progresses_parameter() -> None:
    optimizer = make_optimizer(OptimizerConfig(kind="adam", beta1=0.9, beta2=0.999, epsilon=1e-8))
    current = 1.0
    for _ in range(5):
        current = optimizer.update("p", value=current, grad=1.0, learning_rate=0.1)

    assert current < 1.0
