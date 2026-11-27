#!/usr/bin/env bash
set -euo pipefail
base=/workspace/persist/crivas4-microscopy
exp="$base/experiments/alternative-models"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" LD_PRELOAD="" LD_LIBRARY_PATH="$base/alternative-models-training-env/lib/python3.12/site-packages/torch/lib"
unset PYTHONHOME
export HF_TOKEN_PATH="$base/huggingface-auth/token"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
py="$base/alternative-models-training-env/bin/python"
root="$exp/convnext-paired-fivefold-v1"
mkdir -p "$root"
exec 9>"$exp/.convnext-real-data-smoke.lock"
flock -n 9 || { echo "Another ConvNeXt run is active"; exit 1; }
exec >> "$root/run.log" 2>&1
set -x
for fold in 0 1 2 3 4; do
  ref=$("$py" -c 'import sys,yaml;print(yaml.safe_load(open("configs/train.yaml"))["macenko_reference_patch_ids"][sys.argv[1]])' "$fold")
  for arm in qc_source tumor_roi; do
    dest="$root/fold-$fold/$arm"
    mkdir -p "$dest"
    extra=()
    if [[ "$arm" == tumor_roi ]]; then extra=(--roi-scores "$exp/direct-wsi-roi-v1"); fi
    if [[ ! -f "$dest/prepared.ok" ]]; then
      if [[ -e "$dest/plan" ]]; then mv "$dest/plan" "$dest/plan.interrupted.$(date +%s)"; fi
      "$py" -u -m alternative_models.run prepare --package "$base/packages/AB_qc-v5" --fold "$fold" --all-tiles --seed 0 "${extra[@]}" --out "$dest/plan"
      touch "$dest/prepared.ok"
    fi
    for step in cache train; do
      if [[ -f "$dest/$step.ok" ]]; then continue; fi
      if [[ -e "$dest/$step" ]]; then mv "$dest/$step" "$dest/$step.interrupted.$(date +%s)"; fi
      options=(--plan "$dest/plan/plan.json" --chunk 32 --device cuda:0 --out "$dest/$step")
      if [[ "$step" == cache ]]; then
        options+=(--backbone convnext_v2_t --reference-patch-id "$ref")
      else
        options+=(--cache "$dest/cache" --arm real --tile-chunk 128)
      fi
      "$py" -u -m alternative_models.run "$step" "${options[@]}"
      touch "$dest/$step.ok"
    done
  done
done
"$py" -u -m alternative_models.paired_cv_summary "$root"
