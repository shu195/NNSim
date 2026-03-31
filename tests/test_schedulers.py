from nnsim.schedulers import SchedulerConfig, make_scheduler


def test_constant_scheduler() -> None:
    scheduler = make_scheduler(SchedulerConfig(kind="constant"))
    assert scheduler.get_lr(base_lr=0.5, epoch=1) == 0.5
    assert scheduler.get_lr(base_lr=0.5, epoch=20) == 0.5


def test_exponential_scheduler_decay() -> None:
    scheduler = make_scheduler(SchedulerConfig(kind="exp", gamma=0.9, min_lr=0.01))
    assert abs(scheduler.get_lr(base_lr=1.0, epoch=1) - 1.0) < 1e-12
    assert abs(scheduler.get_lr(base_lr=1.0, epoch=3) - 0.81) < 1e-12


def test_step_scheduler_decay() -> None:
    scheduler = make_scheduler(
        SchedulerConfig(kind="step", gamma=0.5, step_size=2, min_lr=0.01)
    )
    assert abs(scheduler.get_lr(base_lr=1.0, epoch=1) - 1.0) < 1e-12
    assert abs(scheduler.get_lr(base_lr=1.0, epoch=2) - 1.0) < 1e-12
    assert abs(scheduler.get_lr(base_lr=1.0, epoch=3) - 0.5) < 1e-12
