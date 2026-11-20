"""Regression tests for training/tracking.py, with a stand-in for the wandb package."""
import sys
from types import SimpleNamespace

from training import config, tracking


class FakeRun:
    """Keeps what a W&B run is given."""

    def __init__(self, **kwargs):
        self.kwargs, self.logged, self.exit_code = kwargs, [], None

    def log(self, row):
        """Remember a logged row."""
        self.logged.append(row)

    def finish(self, exit_code=0):
        """Remember how the run ended."""
        self.exit_code = exit_code


def fake_wandb(monkeypatch):
    """Install a wandb whose init records its runs; return the list of runs."""
    runs = []
    monkeypatch.setitem(sys.modules, "wandb",
                        SimpleNamespace(init=lambda **kwargs: runs.append(FakeRun(**kwargs))
                                        or runs[-1]))
    return runs


def settings_with(*overrides):
    """Config settings with the given overrides."""
    return config.load_config(backbone="swin_v2_t", overrides=list(overrides))[0]


def test_wandb_is_off_without_a_project(monkeypatch):
    """The default config must not start a run, even when a key is set."""
    runs = fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    assert tracking.start(settings_with(), "run", 0) is None and not runs


def test_wandb_is_off_without_a_key_even_with_a_project(monkeypatch):
    """A project alone does not log: the key must be configured as well."""
    runs = fake_wandb(monkeypatch)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert tracking.start(settings_with("wandb_project=gbm"), "run", 0) is None and not runs


def test_a_run_gets_its_project_name_fold_and_settings_without_paths(monkeypatch):
    """Paths and the worker count stay out of what W&B receives."""
    runs = fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    run = tracking.start(settings_with("wandb_project=gbm", "seed=4"), "swin_fold2", 2)
    assert run is runs[0] and run.kwargs["project"] == "gbm" and run.kwargs["name"] == "swin_fold2"
    sent = run.kwargs["config"]
    assert sent["fold"] == 2 and sent["seed"] == 4 and sent["stages_real"]
    assert not set(tracking.PRIVATE_KEYS) & set(sent)


def test_blank_values_are_not_logged(monkeypatch):
    """The final fit has no validation numbers; an empty value must not reach W&B as a metric."""
    fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    run = tracking.start(settings_with("wandb_project=gbm"), "run", "final")
    tracking.log(run, {"epoch": 1, "train_loss": 0.5, "validation_loss": "", "peak_memory_mb": ""})
    assert run.logged == [{"epoch": 1, "train_loss": 0.5}]


def test_a_failed_run_is_closed_with_a_failing_exit_code():
    """W&B shows the run as crashed, not finished."""
    run = FakeRun()
    tracking.finish(run, failed=True)
    assert run.exit_code == 1
