#!/usr/bin/env bash
set -euo pipefail

BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
GPU_ENV="$BASE/alternative-models-gpu-env"
GPU_PY="$GPU_ENV/bin/python"

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export PIP_CONFIG_FILE=/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ ! -x "$GPU_PY" ]; then
  "$BASE/patch-env/bin/python" -m venv "$GPU_ENV"
fi

"$GPU_PY" -m pip install \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128

"$GPU_PY" -m pip install PyYAML timm

export LD_LIBRARY_PATH="$GPU_ENV/lib/python3.12/site-packages/torch/lib"

"$GPU_PY" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable in this workload"

print("GPU:", torch.cuda.get_device_name(0))
print("Visible GPU memory:",
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
      "GiB")

value = (torch.ones(1, device="cuda") + 1).item()
torch.cuda.synchronize()
print("CUDA execution check:", value)
PY

"$GPU_PY" -m pip freeze \
  > "$ALT/alternative_models/benchmark-environment.lock"
