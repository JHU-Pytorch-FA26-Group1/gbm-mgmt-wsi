#!/usr/bin/env bash
# Separate full data environment; existing benchmark and generation envs are untouched.
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ENV="$BASE/alternative-models-training-env"
TRAIN_PY="$TRAIN_ENV/bin/python"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export PIP_CONFIG_FILE=/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ ! -x "$TRAIN_PY" ]; then
  "$BASE/patch-env/bin/python" -m venv "$TRAIN_ENV"
fi
"$TRAIN_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"'
"$TRAIN_PY" -m pip install "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128
"$TRAIN_PY" -m pip install --constraint "$REPO/alternative_models/training-gpu-constraints.txt" \
  "tiatoolbox==2.1.3" "timm==1.0.27" "h5py==3.16.0" "scikit-learn==1.8.0" \
  "pandas==3.0.2" "numpy==2.4.3" "Pillow==12.1.1" "PyYAML==6.0.2"
"$TRAIN_PY" -m pip check
export LD_LIBRARY_PATH="$TRAIN_ENV/lib/python3.12/site-packages/torch/lib"
"$TRAIN_PY" - <<'PY'
import torch, torchvision, timm, pandas, h5py, sklearn
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
assert torch.__version__ == "2.8.0+cu128"
assert torchvision.__version__ == "0.23.0+cu128"
assert torch.cuda.is_available(), "Run this in the quarter-GPU workspace"
value = (torch.ones(1, device="cuda") + 1).item()
torch.cuda.synchronize()
print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_test": value,
       "visible_memory_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
       "data_imports": "passed", "timm": timm.__version__})
PY
RECORD="$BASE/experiments/alternative-models/training-environment"
mkdir -p "$RECORD"
"$TRAIN_PY" -m pip freeze > "$RECORD/$(date -u +%Y%m%d-%H%M%S)-$$.lock"
