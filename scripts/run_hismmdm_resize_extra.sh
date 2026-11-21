#!/usr/bin/env bash
# Additional NEW resize sources, on a finite 0.5-GPU job; no existing run is overwritten.
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
TEAM="$BASE/gbm-mgmt-wsi"
BATCH="${HIS_RESIZE_EXTRA_BATCH:-$BASE/GBM_MGMT/hismmdm-pilot/fold-0-resize-newtiles-4000}"
CODE="${HIS_RESIZE_EXTRA_CODE:-$TEAM/experiments/hismmdm-pilot}"
HIS_PY="$BASE/hismmdm-env/bin/python"
export PATH="$BASE/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"

infer() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$HIS_PY" -u "$@"
}

case "${1:-}" in
  prepare)
    PATCH_TORCH_LIB="$BASE/patch-env/lib/python3.12/site-packages/torch/lib"
    env LD_LIBRARY_PATH="$PATCH_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" LD_PRELOAD= \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$BASE/patch-env/bin/python" "$CODE/prepare_resize_extension.py" prepare \
      --repo "$TEAM" --root "$BASE/GBM_MGMT" --package "$BASE/packages/A_qc-v5" \
      --labels "$BASE/GBM_MGMT/tcga_mgmt_labels.csv" \
      --parent "$BASE/GBM_MGMT/hismmdm-pilot/fold-0-balanced-4000" --out "$BATCH"
    ;;
  submit)
    infer "$CODE/prepare_resize_extension.py" validate --batch "$BATCH"
    "$HIS_PY" - "$BATCH" "$CODE" <<'PY'
import hashlib, json, sys
from pathlib import Path
batch, code = map(Path, sys.argv[1:])
plan = json.loads((batch / 'batch_plan.json').read_text())
r = json.loads((batch / 'extension_receipt.json').read_text())
if plan['synthetic_count'] != 4000 or plan['source_count'] != 2000 or r['source_overlap'] != 0:
    raise SystemExit('Expected exactly 2,000 new sources / 4,000 variants with zero old-source overlap.')
for name, key in [('prepare_resize_extension.py', 'preparer_sha256'),
                  ('his_mmdm_batch.py', 'legacy_batch_sha256'),
                  ('his_mmdm_he_augment.py', 'shared_core_sha256')]:
    actual = hashlib.sha256((code / name).read_bytes()).hexdigest()
    if actual != r[key]:
        raise SystemExit(f'Code changed since preparation: {name}')
PY
    runai cluster set runai-cluster1
    JOB="hismmdm-resize-extra-$(date -u +%Y%m%d-%H%M%S)"
    RECORD="$BATCH/runai/$JOB"
    mkdir -p "$RECORD/code"
    for FILE in prepare_resize_extension.py his_mmdm_batch.py his_mmdm_he_augment.py; do
      cp "$CODE/$FILE" "$RECORD/code/$FILE"
    done
    cp "${BASH_SOURCE[0]}" "$RECORD/code/run_hismmdm_resize_extra.sh"
    cp "$CODE/RESIZE_EXTRA.md" "$RECORD/code/"
    printf '%s\n' "$JOB" > "$RECORD/job-name.txt"
    runai version > "$RECORD/cli-version.txt"
    git -C "$TEAM" rev-parse HEAD > "$RECORD/team-commit.txt"
    git -C "$BASE/His-MMDM" rev-parse HEAD > "$RECORD/upstream-commit.txt"
    (cd "$RECORD/code"; sha256sum *.py *.sh *.md > SHA256SUMS)
    runai training standard submit "$JOB" --project runai-users \
      --image nvcr.io/nvidia/pytorch:25.06-py3 \
      --gpu-devices-request 1 --gpu-request-type portion \
      --gpu-portion-request 0.5 --gpu-portion-limit 0.5 \
      --cpu-core-request 4 --cpu-memory-request 16G \
      --existing-pvc 'claimname=persistant-datasource-runai-cluster1-f06e9,path=/workspace/persist' \
      --environment-variable "HIS_RESIZE_EXTRA_BATCH=$BATCH" \
      --environment-variable "HIS_RESIZE_EXTRA_CODE=$RECORD/code" \
      --environment-variable "HIS_RESIZE_EXTRA_RECORD=$RECORD" \
      --auto-deletion-time-after-completion 10m --backoff-limit 0 --restart-policy Never \
      --command -- timeout --kill-after=2m 94h \
      bash "$RECORD/code/run_hismmdm_resize_extra.sh" worker \
      2>&1 | tee "$RECORD/submission.txt"
    printf '\nJob: %s\nPersistent records: %s\n' "$JOB" "$RECORD"
    ;;
  worker)
    : "${HIS_RESIZE_EXTRA_RECORD:?Use submit to launch a recorded finite job}"
    exec > >(tee -a "$HIS_RESIZE_EXTRA_RECORD/runtime.log") 2>&1
    infer "$CODE/prepare_resize_extension.py" validate --batch "$BATCH"
    infer -c 'import json,torch; assert torch.cuda.is_available(); print(json.dumps({"gpu":torch.cuda.get_device_name(0),"visible_memory_GiB":torch.cuda.get_device_properties(0).total_memory/2**30,"torch":torch.__version__,"cuda":torch.version.cuda}))'
    for SHARD in 0 1; do
      infer "$CODE/his_mmdm_batch.py" worker --batch "$BATCH" --shard "$SHARD" \
        --upstream "$BASE/His-MMDM" --weights "$BASE/GBM_MGMT/hismmdm-pilot/artifacts/hmu-1st" --device cuda:0
    done
    infer "$CODE/his_mmdm_batch.py" verify --batch "$BATCH"
    ;;
  status|verify)
    infer "$CODE/his_mmdm_batch.py" "$1" --batch "$BATCH"
    ;;
  *)
    printf 'Usage: bash scripts/run_hismmdm_resize_extra.sh prepare|submit|status|verify\n'
    exit 2
    ;;
esac
