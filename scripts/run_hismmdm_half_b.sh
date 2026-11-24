#!/usr/bin/env bash
# CPU preparation, separately submitted finite crop/resize jobs; no automatic submission.
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
REPO="${HIS_B_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE="${HIS_B_CODE:-$REPO}"
ROOT="$BASE/GBM_MGMT/hismmdm-pilot/half-B-fold-0-matched-32000"
export PATH="$BASE/bin:/root/.runai/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"

cpu() {
  env PYTHONPATH="$CODE" LD_PRELOAD="" \
    LD_LIBRARY_PATH="$BASE/patch-env/lib/python3.12/site-packages/torch/lib" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$BASE/patch-env/bin/python" -u -m alternative_models.half_b_generation \
    "$@" --base "$BASE" --repo "$REPO"
}

infer() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONHOME PYTHONPATH="$CODE" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$BASE/hismmdm-env/bin/python" -u -m alternative_models.half_b_generation \
    "$@" --base "$BASE" --repo "$REPO"
}

MODE="${1:-}"
case "$MODE" in
  prepare|validate|status|a-status|finish-crops|record)
    cpu "$MODE"
    ;;
  submit-crop|submit-resize)
    ARM="${MODE#submit-}"
    cpu validate
    cpu a-complete --arm "$ARM"
    runai cluster set runai-cluster1
    mkdir -p "$ROOT/runai"
    exec 9> "$ROOT/.submit.lock"
    flock -n 9 || { echo 'Another Half B submission is in progress.'; exit 1; }
    TABLE="$(mktemp "$ROOT/runai/.scheduler.XXXXXX")"
    trap 'rm -f "$TABLE"' EXIT
    runai training standard list --project runai-users > "$TABLE"
    cpu scheduler-gate --arm "$ARM" --scheduler-table "$TABLE"
    JOB="hismmdm-halfb-$ARM-$(date -u +%Y%m%d-%H%M%S)"
    RECORD="$ROOT/runai/$JOB"
    mkdir -p "$RECORD/code/alternative_models" "$RECORD/code/scripts"
    cp "$TABLE" "$RECORD/workloads-before.txt"
    for FILE in __init__.py half_b_generation.py half_b_package.py; do
      cp "$REPO/alternative_models/$FILE" "$RECORD/code/alternative_models/$FILE"
    done
    cp "${BASH_SOURCE[0]}" "$RECORD/code/scripts/run_hismmdm_half_b.sh"
    printf '%s\n' "$JOB" > "$RECORD/job-name.txt"
    runai version > "$RECORD/cli-version.txt"
    git -C "$REPO" rev-parse HEAD > "$RECORD/repo-commit.txt"
    git -C "$REPO" status --short -- alternative_models scripts/run_hismmdm_half_b.sh > "$RECORD/code-status.txt"
    git -C "$BASE/His-MMDM" rev-parse HEAD > "$RECORD/upstream-commit.txt"
    (cd "$RECORD/code"; sha256sum alternative_models/*.py scripts/*.sh > SHA256SUMS)
    PORTION=0.5
    [ "$ARM" != crop ] || PORTION=1
    COMMAND=(runai training standard submit "$JOB" --project runai-users
      --image nvcr.io/nvidia/pytorch:25.06-py3
      --gpu-devices-request 1 --gpu-request-type portion --gpu-portion-request "$PORTION"
      --cpu-core-request 4 --cpu-memory-request 16G
      --existing-pvc 'claimname=persistant-datasource-runai-cluster1-f06e9,path=/workspace/persist'
      --environment-variable "HIS_B_CODE=$RECORD/code"
      --environment-variable "HIS_B_REPO=$REPO"
      --environment-variable "HIS_B_RECORD=$RECORD"
      --environment-variable "HIS_B_ARM=$ARM"
      --auto-deletion-time-after-completion 10m --backoff-limit 0 --restart-policy Never
      --command -- timeout --kill-after=2m 94h bash "$RECORD/code/scripts/run_hismmdm_half_b.sh" worker)
    printf '%q ' "${COMMAND[@]}" > "$RECORD/submit-command.sh"
    printf '\n' >> "$RECORD/submit-command.sh"
    "${COMMAND[@]}" 2>&1 | tee "$RECORD/submission.txt"
    printf '\nJob: %s\nPersistent records: %s\n' "$JOB" "$RECORD"
    ;;
  worker)
    : "${HIS_B_RECORD:?Run through a recorded submission}"
    : "${HIS_B_ARM:?Missing design}"
    exec > >(tee -a "$HIS_B_RECORD/runtime.log") 2>&1
    infer worker --arm "$HIS_B_ARM"
    # Same already-approved blur-10 screening, on the same 224-pixel copies.
    cpu qc --arm "$HIS_B_ARM"
    ;;
  qc-crop|qc-resize)
    cpu qc --arm "${MODE#qc-}"
    ;;
  *)
    echo 'Usage: bash scripts/run_hismmdm_half_b.sh prepare|validate|a-status|status|finish-crops|record|submit-crop|submit-resize|qc-crop|qc-resize'
    exit 2
    ;;
esac
