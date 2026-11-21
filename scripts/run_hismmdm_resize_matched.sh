#!/usr/bin/env bash
# Same originals; original 4,000 + six new variants per original = 16,000 resize outputs.
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
TEAM="$BASE/gbm-mgmt-wsi"
BATCH="$BASE/GBM_MGMT/hismmdm-pilot/fold-0-resize-matched-additional12000"
CODE="${HIS_MATCHED_CODE:-$TEAM/experiments/hismmdm-pilot}"
PY="$BASE/hismmdm-env/bin/python"
export PATH="$BASE/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"

infer() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" -u "$@"
}

case "${1:-}" in
  prepare)
    infer "$CODE/his_mmdm_resize_matched.py" prepare \
      --parent "$BASE/GBM_MGMT/hismmdm-pilot/fold-0-balanced-4000" \
      --crop "$BASE/GBM_MGMT/hismmdm-pilot/fold-0-corners-16000" --out "$BATCH"
    ;;
  submit)
    infer "$CODE/his_mmdm_resize_matched.py" validate --batch "$BATCH"
    "$PY" - "$BATCH" "$CODE" <<'PY'
import hashlib, json, sys
from pathlib import Path
batch, code = map(Path, sys.argv[1:])
p = json.loads((batch/'batch_plan.json').read_text())
if (p['source_count'], p['existing_outputs'], p['synthetic_count'], p['matched_total']) != (2000, 4000, 12000, 16000):
    raise SystemExit('Expected EXACTLY 2,000 original tiles / 4,000 retained + 12,000 new = 16,000.')
for name, key in [('his_mmdm_resize_matched.py', 'preparer_sha256'),
                  ('his_mmdm_batch.py', 'legacy_sha256'), ('his_mmdm_he_augment.py', 'core_sha256')]:
    if hashlib.sha256((code/name).read_bytes()).hexdigest() != p[key]:
        raise SystemExit(f'Code changed since preparation: {name}')
PY
    runai cluster set runai-cluster1
    JOB="hismmdm-resize-matched12k-$(date -u +%Y%m%d-%H%M%S)"
    RECORD="$BATCH/runai/$JOB"
    mkdir -p "$RECORD/code"
    for FILE in his_mmdm_resize_matched.py his_mmdm_batch.py his_mmdm_he_augment.py RESIZE_MATCHED.md; do
      cp "$CODE/$FILE" "$RECORD/code/$FILE"
    done
    cp "${BASH_SOURCE[0]}" "$RECORD/code/run_hismmdm_resize_matched.sh"
    printf '%s\n' "$JOB" > "$RECORD/job-name.txt"
    runai version > "$RECORD/cli-version.txt"
    git -C "$TEAM" rev-parse HEAD > "$RECORD/team-commit.txt"
    git -C "$BASE/His-MMDM" rev-parse HEAD > "$RECORD/upstream-commit.txt"
    (cd "$RECORD/code"; sha256sum *.py *.sh *.md > SHA256SUMS)
    runai training standard submit "$JOB" --project runai-users \
      --image nvcr.io/nvidia/pytorch:25.06-py3 \
      --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request 0.5 \
      --cpu-core-request 4 --cpu-memory-request 16G \
      --existing-pvc 'claimname=persistant-datasource-runai-cluster1-f06e9,path=/workspace/persist' \
      --environment-variable "HIS_MATCHED_CODE=$RECORD/code" \
      --environment-variable "HIS_MATCHED_RECORD=$RECORD" \
      --auto-deletion-time-after-completion 10m --backoff-limit 0 --restart-policy Never \
      --command -- timeout --kill-after=2m 94h \
      bash "$RECORD/code/run_hismmdm_resize_matched.sh" worker \
      2>&1 | tee "$RECORD/submission.txt"
    printf '\nJob: %s\nRecords: %s\n' "$JOB" "$RECORD"
    ;;
  worker)
    : "${HIS_MATCHED_RECORD:?Use submit for a recorded finite job}"
    exec 9> "$BATCH/.job.lock"
    flock -n 9 || { printf 'Another matched-resize worker is already active.\n'; exit 1; }
    exec > >(tee -a "$HIS_MATCHED_RECORD/runtime.log") 2>&1
    for SHARD in 0 1; do
      infer "$CODE/his_mmdm_resize_matched.py" worker --batch "$BATCH" --shard "$SHARD" \
        --upstream "$BASE/His-MMDM" --weights "$BASE/GBM_MGMT/hismmdm-pilot/artifacts/hmu-1st"
    done
    infer "$CODE/his_mmdm_resize_matched.py" verify --batch "$BATCH"
    ;;
  status|validate|verify)
    infer "$CODE/his_mmdm_resize_matched.py" "$1" --batch "$BATCH"
    ;;
  *) printf 'Usage: bash scripts/run_hismmdm_resize_matched.sh prepare|submit|status|validate|verify\n'; exit 2 ;;
esac
