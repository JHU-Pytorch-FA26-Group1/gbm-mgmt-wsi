#!/usr/bin/env bash
# Review-only dependency resolution. Does not install/uninstall any package.
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
ALT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_PY="$BASE/alternative-models-gpu-env/bin/python"
CONSTRAINTS="$ALT/alternative_models/training-gpu-constraints.txt"
REPORT_DIR="$BASE/experiments/alternative-models/dependency-review"

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export PIP_CONFIG_FILE=/dev/null
export LD_LIBRARY_PATH="$BASE/alternative-models-gpu-env/lib/python3.12/site-packages/torch/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$GPU_PY" - <<'PY'
from importlib.metadata import version
for name, expected in (("torch", "2.8.0+cu128"), ("torchvision", "0.23.0+cu128")):
    actual = version(name)
    if actual != expected:
        raise SystemExit(f"Unexpected existing {name}: {actual}; expected {expected}")
PY

mkdir -p "$REPORT_DIR"
REPORT="$REPORT_DIR/$(date -u +%Y%m%d-%H%M%S)-$$.json"
if ! "$GPU_PY" -m pip install --dry-run --report "$REPORT" \
    --constraint "$CONSTRAINTS" \
    "tiatoolbox==2.1.3" "h5py==3.16.0" "scikit-learn==1.8.0" pandas \
    > "$REPORT.log" 2>&1; then
  tail -n 50 "$REPORT.log"
  printf '\nDependency resolution failed; no installation performed. Log: %s\n' "$REPORT.log"
  exit 1
fi

"$GPU_PY" - "$REPORT" <<'PY'
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = json.loads(path.read_text())
changes = []
for record in report.get("install", []):
    name, target = record["metadata"]["name"], record["metadata"]["version"]
    try:
        current = version(name)
    except PackageNotFoundError:
        current = None
    if current != target:
        changes.append(dict(package=name, current=current, proposed=target))
    if name.lower() in ("torch", "torchvision") and current != target:
        raise SystemExit("Unexpected CUDA package change in dry-run plan; nothing was installed")
existing = [r for r in changes if r["current"] is not None]
required = [r for r in changes if r["package"].lower().replace("_", "-") in
            {"tiatoolbox", "h5py", "scikit-learn", "pandas"}]
print(json.dumps(dict(report=str(path), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    packages_proposed=len(changes), changes_to_existing_packages=existing,
    required_data_packages=required, packages_installed=0,
    note="Review auxiliary-version changes before installation; repeat the GPU benchmark after any backbone dependency change"), indent=2))
PY
