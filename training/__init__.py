"""Training code. Puts data_prep on sys.path: its modules import each other by bare name."""
import sys
from pathlib import Path

_DATA_PREP = str(Path(__file__).resolve().parent.parent / "data_prep")
if _DATA_PREP not in sys.path:
    sys.path.insert(0, _DATA_PREP)
