#!/usr/bin/env bash
# Run the tiling pipeline for one subset: register -> masks -> tile -> audit -> grandqc -> qc -> export.
#
# Usage:
#   scripts/run_pipeline.sh --root ~/Bioinformatics/WSI/data --subset B
#   scripts/run_pipeline.sh --root ~/Bioinformatics/WSI/data --subset B --steps tile,audit --workers 8
#   scripts/run_pipeline.sh --root /workspace/GBM_MGMT --subset A --steps register,tile,audit,qc
#
# Options (or the environment variables in brackets):
#   --root DIR        data folder with raw/ and state/ (required) [GBM_ROOT]
#   --subset S        full, A or B (required) [GBM_SUBSET]
#   --steps LIST      comma-separated, in this order (default: tile,audit,grandqc,qc,export) [GBM_STEPS]
#                     register  verify slides already in raw/ and write receipts (download.py --register)
#                     masks     mask comparison panels only (tile.py --masks-only)
#                     tile      extract tiles (tile.py)
#                     audit     recheck every file and pixel hash (tile.py --audit)
#                     grandqc   per-tile GrandQC fractions from the GrandQC masks (grandqc.py, 1 worker)
#                     qc        apply configs/qc_policy.yaml (qc.py)
#                     export    build modeling_patches.csv (export.py; needs a frozen policy)
#   --policy FILE     QC policy for qc and export; repeat it for one policy per cohort (qc runs once
#                     per policy, export gets them all) [GBM_POLICY, comma-separated]
#   --folds FILE      shared patient fold map for qc and export [GBM_FOLDS]
#   --workers N       slides processed in parallel (default: tile.py's, from the CPU allocation) [GBM_WORKERS]
#   --python CMD      Python to use (default: python from the active environment) [GBM_PYTHON]
#   --dry-run         print the commands without running them
#
# Runs the steps in order and stops at the first one that does not exit 0, passing its exit code
# through (1 = some slides or patients unfinished, 2 = setup problem, 130 = interrupted). Each step
# writes its own log to logs/. Uses the active Python environment; activate gbm-mgmt (or your venv)
# first.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${GBM_ROOT:-}"
subset="${GBM_SUBSET:-}"
steps="${GBM_STEPS:-tile,audit,grandqc,qc,export}"
workers="${GBM_WORKERS:-}"
policies=()
[[ -n "${GBM_POLICY:-}" ]] && IFS=',' read -r -a policies <<< "$GBM_POLICY"
folds="${GBM_FOLDS:-}"
python="${GBM_PYTHON:-python}"
dry_run=0

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) root="$2"; shift 2 ;;
        --subset) subset="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --policy) policies+=("$2"); shift 2 ;;
        --folds) folds="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$root" || -z "$subset" ]]; then
    echo "--root and --subset are required" >&2
    exit 2
fi
case "$subset" in full|A|B) ;; *) echo "--subset must be full, A or B: $subset" >&2; exit 2 ;; esac
if [[ -n "$workers" && ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "--workers must be a positive integer: $workers" >&2
    exit 2
fi

order=(register masks tile audit grandqc qc export)
IFS=',' read -r -a requested <<< "$steps"
for step in "${requested[@]}"; do
    [[ " ${order[*]} " == *" $step "* ]] || { echo "unknown step: $step" >&2; exit 2; }
done

worker_args=()
[[ -n "$workers" ]] && worker_args=(--workers "$workers")
common=(--root "$root" --subset "$subset")
folds_args=()
[[ -n "$folds" ]] && folds_args+=(--folds "$folds")
export_args=()
[[ ${#policies[@]} -gt 0 ]] && export_args+=(--policy "${policies[@]}")

build_command() {   # sets the array `cmd` for one step
    case "$1" in
        register) cmd=("$python" "$repo/data_prep/download.py" "${common[@]}" --register) ;;
        masks) cmd=("$python" "$repo/data_prep/tile.py" "${common[@]}" --masks-only ${worker_args[@]+"${worker_args[@]}"}) ;;
        tile) cmd=("$python" "$repo/data_prep/tile.py" "${common[@]}" ${worker_args[@]+"${worker_args[@]}"}) ;;
        audit) cmd=("$python" "$repo/data_prep/tile.py" "${common[@]}" --audit ${worker_args[@]+"${worker_args[@]}"}) ;;
        grandqc) cmd=("$python" "$repo/data_prep/grandqc.py" "${common[@]}") ;;
        qc) cmd=("$python" "$repo/data_prep/qc.py" "${common[@]}" ${folds_args[@]+"${folds_args[@]}"}
                 ${2:+--policy "$2"}) ;;
        export) cmd=("$python" "$repo/data_prep/export.py" "${common[@]}" ${folds_args[@]+"${folds_args[@]}"}
                     ${export_args[@]+"${export_args[@]}"}) ;;
    esac
}

run_step() {   # build and run one command; exit on failure
    build_command "$@"
    echo "==> $1: ${cmd[*]}"
    [[ $dry_run -eq 1 ]] && return 0
    set +e
    "${cmd[@]}"
    code=$?
    set -e
    if [[ $code -ne 0 ]]; then
        echo "==> $1 stopped with exit code $code; later steps not run." >&2
        exit "$code"
    fi
}

for step in "${order[@]}"; do
    [[ ",$steps," == *",$step,"* ]] || continue
    if [[ $step == qc && ${#policies[@]} -gt 0 ]]; then
        for policy in "${policies[@]}"; do run_step qc "$policy"; done
    else
        run_step "$step"
    fi
done
echo "==> done: ${steps}"
