#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
ROI_ENV="$BASE/alternative-models-roi-env"
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export PIP_CONFIG_FILE=/dev/null
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ ! -x "$ROI_ENV/bin/python" ]; then
  ROI_BOOTSTRAP_PY="${ROI_BOOTSTRAP_PY:-$BASE/patch-env/bin/python}"
  if [ ! -x "$ROI_BOOTSTRAP_PY" ]; then ROI_BOOTSTRAP_PY="$(command -v python3)"; fi
  "$ROI_BOOTSTRAP_PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10--3.12 for the pinned ROI environment"'
  "$ROI_BOOTSTRAP_PY" -m venv "$ROI_ENV"
fi
"$ROI_ENV/bin/python" -m pip install \
  'numpy==1.26.4' 'Pillow==11.3.0' 'h5py==3.16.0' \
  'onnx==1.17.0' 'onnxruntime==1.22.0' 'torchstain==1.3.0'
"$ROI_ENV/bin/python" -m pip check
"$ROI_ENV/bin/python" -m pip freeze > "$ROI_ENV/environment.lock"
echo "CPU-only ROI environment ready: $ROI_ENV"
