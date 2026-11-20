#!/usr/bin/env bash
# Finite foreground jobs; persistent receipts; never launches the old resize arm.
set -euo pipefail
BASE="${BASE:-/workspace/persist/crivas4-microscopy}"
TEAM="$BASE/gbm-mgmt-wsi"
BATCH="${HIS_CORNERS_BATCH:-$BASE/GBM_MGMT/hismmdm-pilot/fold-0-corners-16000}"
SCRIPT="${HIS_CORNERS_SCRIPT:-$TEAM/experiments/hismmdm-pilot/his_mmdm_corners.py}"
PY="$BASE/hismmdm-env/bin/python"
export PATH="$BASE/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"
MODE="${1:-}"

infer() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" -u "$SCRIPT" "$@"
}

case "$MODE" in
  prepare)
    env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" "$SCRIPT" prepare \
      --parent "$BASE/GBM_MGMT/hismmdm-pilot/fold-0-balanced-4000" --out "$BATCH"
    ;;
  status|verify)
    infer "$MODE" --batch "$BATCH"
    ;;
  benchmark-worker|worker)
    : "${HIS_RUN_DIR:?Run through submit-benchmark or submit-run}"
    mkdir -p "$HIS_RUN_DIR"
    exec > >(tee -a "$HIS_RUN_DIR/runtime.log") 2>&1
    if [ "$MODE" = benchmark-worker ]; then
      for SIZE in ${HIS_BENCHMARK_SIZES:-4 8}; do
        infer worker --batch "$BATCH" --upstream "$BASE/His-MMDM" \
          --weights "$BASE/GBM_MGMT/hismmdm-pilot/artifacts/hmu-1st" \
          --shard 0 --batch-size "$SIZE" --limit-groups 2 \
          --benchmark-out "$BATCH/benchmarks/$(basename "$HIS_RUN_DIR")/bs$SIZE"
      done
    else
      : "${HIS_BATCH_SIZE:?Select a benchmarked batch size}"
      for SHARD in 0 1; do
        infer worker --batch "$BATCH" --upstream "$BASE/His-MMDM" \
          --weights "$BASE/GBM_MGMT/hismmdm-pilot/artifacts/hmu-1st" \
          --shard "$SHARD" --batch-size "$HIS_BATCH_SIZE"
      done
      infer verify --batch "$BATCH"
    fi
    ;;
  submit-benchmark|submit-run)
    SIZE="${2:-8}"
    case "$SIZE" in 4|8|16) ;; *) printf 'Batch size must be 4, 8 or 16.\n'; exit 2 ;; esac
    BENCHMARK_SIZES="${2:-4 8}"
    test -s "$BATCH/batch_plan.json"
    if [ "$MODE" = submit-run ]; then
      # Resume is permitted with a matching prior benchmark; changing the size is
      # rejected later if production receipts already use another size.
      "$PY" - "$BATCH" "$SIZE" "$SCRIPT" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
batch, size, script = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
matches = []
for path in (batch / 'benchmarks').glob(f'*/bs{size}/generated/shard-0/completed.json'):
    r = json.loads(path.read_text())
    saved_run = json.loads((path.parent / 'batch_run.json').read_text())
    run = saved_run['signature']
    signature_hash = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    if (r.get('phase') == 'benchmark' and r.get('batch_size') == size
        and r.get('outputs', 0) >= 4 * size
        and signature_hash == saved_run['signature_sha256'] == r['signature_sha256']
        and digest(path.parent / 'generation_manifest.csv') == r['manifest_sha256']
        and run['runner_sha256'] == digest(script)
        and run['wrapper_sha256'] == digest(script.parent / 'his_mmdm_he_augment.py')
        and run['batch_plan_sha256'] == digest(batch / 'batch_plan.json')):
        matches.append((path, r))
if not matches:
    raise SystemExit('No completed benchmark matches this batch size and current code/inputs.')
path, r = max(matches, key=lambda pair: pair[0].stat().st_mtime)
hours = r['estimated_full_generation_hours']
if not math.isfinite(hours) or not 0 < hours <= 90:
    raise SystemExit(f'Estimate {hours:.1f} h exceeds the 90 h launch gate; inspect before launching.')
print(f'Benchmark: {path}\nEstimated generation: {hours:.1f} h; queue/preemption time excluded.')
PY
    fi
    JOB="hismmdm-corners-$([ "$MODE" = submit-benchmark ] && printf bench || printf run)-$(date -u +%Y%m%d-%H%M%S)"
    RECORD="$BATCH/runai/$JOB"
    mkdir -p "$RECORD/code"
    cp "$SCRIPT" "$RECORD/code/his_mmdm_corners.py"
    cp "$(dirname "$SCRIPT")/his_mmdm_he_augment.py" "$RECORD/code/"
    cp "${BASH_SOURCE[0]}" "$RECORD/code/run_hismmdm_corners.sh"
    printf '%s\n' "$JOB" > "$RECORD/job-name.txt"
    runai version > "$RECORD/cli-version.txt"
    git -C "$TEAM" rev-parse HEAD > "$RECORD/team-commit.txt"
    git -C "$BASE/His-MMDM" rev-parse HEAD > "$RECORD/upstream-commit.txt"
    (cd "$RECORD/code"; sha256sum *.py *.sh > SHA256SUMS)
    if [ "$MODE" = submit-benchmark ]; then
      ACTION=benchmark-worker
      DURATION=2h
    else
      ACTION=worker
      DURATION=94h
    fi
    # An existing PVC survives automatic deletion of the completed job.
    runai training standard submit "$JOB" --project runai-users \
      --image nvcr.io/nvidia/pytorch:25.06-py3 \
      --gpu-devices-request 1 --gpu-request-type portion \
      --gpu-portion-request 1 --gpu-portion-limit 1 \
      --cpu-core-request 4 --cpu-memory-request 16G \
      --existing-pvc 'claimname=persistant-datasource-runai-cluster1-f06e9,path=/workspace/persist' \
      --environment-variable "HIS_CORNERS_BATCH=$BATCH" \
      --environment-variable "HIS_CORNERS_SCRIPT=$RECORD/code/his_mmdm_corners.py" \
      --environment-variable "HIS_RUN_DIR=$RECORD" \
      --environment-variable "HIS_BATCH_SIZE=$SIZE" \
      --environment-variable "HIS_BENCHMARK_SIZES=$BENCHMARK_SIZES" \
      --auto-deletion-time-after-completion 10m --backoff-limit 0 --restart-policy Never \
      --command -- timeout --kill-after=2m "$DURATION" \
      bash "$RECORD/code/run_hismmdm_corners.sh" "$ACTION" \
      2>&1 | tee "$RECORD/submission.txt"
    printf '\nJob: %s\nPersistent records: %s\n' "$JOB" "$RECORD"
    ;;
  *)
    printf 'Usage: bash scripts/run_hismmdm_corners.sh prepare|submit-benchmark|submit-run SIZE|status|verify\n'
    exit 2
    ;;
esac
