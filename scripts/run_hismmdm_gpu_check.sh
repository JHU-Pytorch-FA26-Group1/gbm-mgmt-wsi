#!/usr/bin/env bash
set -euo pipefail

BASE=/workspace/persist/crivas4-microscopy
export PATH="$BASE/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"

JOB="hismmdm-gpu-check-$(date -u +%Y%m%d-%H%M%S)"
RECORD="$BASE/GBM_MGMT/hismmdm-pilot/runai/$JOB"
mkdir -p "$RECORD"

printf '%s\n' "$JOB" > "$RECORD/job-name.txt"
runai version > "$RECORD/cli-version.txt"

runai training standard submit "$JOB" \
  --project runai-users \
  --image nvcr.io/nvidia/pytorch:25.06-py3 \
  --gpu-devices-request 1 \
  --gpu-request-type portion \
  --gpu-portion-request 1 \
  --gpu-portion-limit 1 \
  --cpu-core-request 4 \
  --cpu-memory-request 16G \
  --existing-pvc \
    "claimname=persistant-datasource-runai-cluster1-f06e9,path=/workspace/persist" \
  --auto-deletion-time-after-completion 10m \
  --backoff-limit 0 \
  --restart-policy Never \
  --command -- timeout 10m \
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$BASE/hismmdm-env/bin/python" -c '
import json, platform
from pathlib import Path
import torch

base = Path("/workspace/persist/crivas4-microscopy")
required = [
    base / "His-MMDM",
    base / "packages/A_qc-v5/package_manifest.json",
    base / "GBM_MGMT/hismmdm-pilot/fold-0-balanced-4000/batch_plan.json",
    base / "GBM_MGMT/hismmdm-pilot/artifacts/hmu-1st/diffusion_model308000.pt",
]
for path in required:
    if not path.exists():
        raise SystemExit(f"Missing from mounted storage: {path}")

assert torch.cuda.is_available(), "CUDA is unavailable"
value = (torch.ones(1, device="cuda") + 1).item()
torch.cuda.synchronize()

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "visible_gpu_memory_GiB": round(
        torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    "cuda_test": value,
    "required_paths_accessible": True,
}, indent=2))
' 2>&1 | tee "$RECORD/submission.txt"

printf '\nJob: %s\nRecords: %s\n' "$JOB" "$RECORD"
