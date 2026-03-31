from __future__ import annotations

from dataclasses import dataclass, field


class Optimizer:
    def update(self, key: str, value: float, grad: float, learning_rate: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str = "sgd"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass
class SGDOptimizer(Optimizer):
    def update(self, key: str, value: float, grad: float, learning_rate: float) -> float:
        del key
        return value - learning_rate * grad


@dataclass
class MomentumOptimizer(Optimizer):
    momentum: float
    velocity: dict[str, float] = field(default_factory=dict)

    def update(self, key: str, value: float, grad: float, learning_rate: float) -> float:
        previous = self.velocity.get(key, 0.0)
        current = self.momentum * previous + (1.0 - self.momentum) * grad
        self.velocity[key] = current
        return value - learning_rate * current


@dataclass
class AdamOptimizer(Optimizer):
    beta1: float
    beta2: float
    epsilon: float
    m: dict[str, float] = field(default_factory=dict)
    v: dict[str, float] = field(default_factory=dict)
    t: dict[str, int] = field(default_factory=dict)

    def update(self, key: str, value: float, grad: float, learning_rate: float) -> float:
        t = self.t.get(key, 0) + 1
        m_prev = self.m.get(key, 0.0)
        v_prev = self.v.get(key, 0.0)

        m_cur = self.beta1 * m_prev + (1.0 - self.beta1) * grad
        v_cur = self.beta2 * v_prev + (1.0 - self.beta2) * (grad * grad)

        m_hat = m_cur / (1.0 - (self.beta1 ** t))
        v_hat = v_cur / (1.0 - (self.beta2 ** t))

        self.t[key] = t
        self.m[key] = m_cur
        self.v[key] = v_cur

        return float(value - learning_rate * m_hat / ((v_hat ** 0.5) + self.epsilon))


def make_optimizer(config: OptimizerConfig) -> Optimizer:
    name = config.kind.lower()
    if name == "sgd":
        return SGDOptimizer()
    if name == "momentum":
        return MomentumOptimizer(momentum=config.momentum)
    if name == "adam":
        return AdamOptimizer(
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
        )
    raise ValueError("Unknown optimizer kind. Use sgd, momentum, or adam")