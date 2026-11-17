"""Put the repo root on sys.path so `import training` works."""
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
