from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    kind: str = "constant"
    gamma: float = 0.95
    step_size: int = 50
    min_lr: float = 1e-5


class LearningRateScheduler:
    def get_lr(self, base_lr: float, epoch: int) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantLRScheduler(LearningRateScheduler):
    def get_lr(self, base_lr: float, epoch: int) -> float:
        del epoch
        return base_lr


@dataclass(frozen=True)
class ExponentialDecayScheduler(LearningRateScheduler):
    gamma: float
    min_lr: float

    def get_lr(self, base_lr: float, epoch: int) -> float:
        lr = base_lr * (self.gamma ** max(0, epoch - 1))
        return max(self.min_lr, lr)


@dataclass(frozen=True)
class StepDecayScheduler(LearningRateScheduler):
    gamma: float
    step_size: int
    min_lr: float

    def get_lr(self, base_lr: float, epoch: int) -> float:
        if self.step_size < 1:
            raise ValueError("step_size must be >= 1")
        step_index = (max(1, epoch) - 1) // self.step_size
        lr = base_lr * (self.gamma ** step_index)
        return max(self.min_lr, lr)


def make_scheduler(config: SchedulerConfig) -> LearningRateScheduler:
    name = config.kind.lower()
    if name == "constant":
        return ConstantLRScheduler()
    if name == "exp":
        return ExponentialDecayScheduler(gamma=config.gamma, min_lr=config.min_lr)
    if name == "step":
        return StepDecayScheduler(
            gamma=config.gamma,
            step_size=config.step_size,
            min_lr=config.min_lr,
        )

    raise ValueError("Unknown scheduler kind. Use constant, exp, or step")
