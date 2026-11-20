"""Put the repo root on sys.path so `import training` works; keep tests off any real W&B account."""
import sys
from pathlib import Path

import pytest

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def no_wandb(monkeypatch):
    """Whatever the shell has configured, no test reaches a real W&B account."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
