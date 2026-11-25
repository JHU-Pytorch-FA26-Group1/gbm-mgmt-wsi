# Next step: real A+B data, ConvNeXt first

Prerequisite: the whole-patient v2 patch is already applied and its quarter-GPU
mathematical smoke passed. This patch adds only files under `alternative_models/`.
It neither submits a Run:ai job nor changes generation, QC, owner models or folds.

## Why a separate environment

The CUDA benchmark environment lacks several data dependencies. TIAToolbox's
compatible dependency versions differ from the original benchmark versions.
`setup_training_env.sh` creates `alternative-models-training-env` under the
persistent base. It installs the same verified torch 2.8.0+cu128 / torchvision
0.23.0+cu128 builds and compatible data packages, checks imports and CUDA, and
records `pip freeze`. It preserves the benchmark, patch and diffusion environments.
The installer uses normal dependency resolution with CUDA constraints, not
`--no-deps`; inherited NVIDIA pip constraints are cleared inside the script.
TIAToolbox 2.1.3 includes an OpenSlide binary dependency, so no sudo install is
requested. This is a one-time environment setup, not a training run.

## What the diagnostic does

1. Use the existing owner combiner to create a complete A+B index if absent.
   Tile payloads remain in their original packages; no image copy occurs.
2. Check manifest compatibility, source paths, canonical patient identities,
   fold/label/half consistency, duplicate tile IDs and package completeness.
   Inventory all TCGA and CPTAC metadata; list patients with no model tiles.
3. Select one TCGA patient for each half/class/role combination: four training
   and four development patients, both classes represented in both A and B.
4. Select at most 129 distinct real tiles per diagnostic patient. This deliberate
   smoke-test subset is recorded, never reported as full-patient classification.
5. Load the actual pretrained ConvNeXt V2-T weights. Fit the existing Macenko
   transform on the existing configured fold-specific training reference tile;
   the owner helper refuses a held-out or CPTAC reference. Use the unchanged
   verified HDF5/PNG reader, image pixel hashes and ImageNet tensor preprocessing.
6. Run one frozen-head epoch, then one final-stage fine-tuning epoch, with one
   global patient loss and update per declared smoke pool. Fine-tuning chunks
   default to 128 and retain any partial final chunk. Earlier encoder stages
   remain frozen. The test rereads real pixels for the gradient replay.
7. Produce patient predictions, accuracy, confusion counts, AUROC, checkpoints,
   dependency/code/input hashes, stage timings and peak GPU memory. Re-encode
   development tiles after fine-tuning; never reuse stale frozen features.

Every output is marked `valid_for_results: false`. The purpose is to uncover data,
preprocessing, pretrained-download, gradient, memory or I/O problems before long
jobs. Neither four development patients nor a 129-tile subset can establish model
quality. The subsequent real-only A+B experiment uses the full declared patient
pools and the agreed 200-epoch maximum/plateau rule. Inner selection and independent
outer testing remain separate steps; CPTAC pixels are never read by this smoke.
MRI is outside this work until the team decides to pursue it.

## Apply, install, inventory and run

Download `alternative-models-real-data-smoke.zip` into the persistent base folder.
The environment installation can take several minutes; normal pip progress will
be visible. No password or new Run:ai sign-in is needed.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"

(
set -euo pipefail
python3 -m zipfile -e "$BASE/alternative-models-real-data-smoke.zip" "$BASE/real-data-smoke"
cd "$ALT"
PATCH="$BASE/real-data-smoke/alternative-models-real-data-smoke.patch"
git apply --check "$PATCH"
git apply "$PATCH"
bash -n alternative_models/setup_training_env.sh
bash alternative_models/setup_training_env.sh
)
```

Then run the metadata inventory and the bounded diagnostics on the existing
quarter-GPU workspace. New output paths are required; failed receipts are retained.
The timestamp avoids overwriting an earlier attempt.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
TRAIN_PY="$BASE/alternative-models-training-env/bin/python"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="$BASE/experiments/alternative-models/real-data-smoke-$STAMP"

(
set -euo pipefail
cd "$ALT"
unset PYTHONPATH PYTHONHOME
export LD_PRELOAD=""
export LD_LIBRARY_PATH="$BASE/alternative-models-training-env/lib/python3.12/site-packages/torch/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$TRAIN_PY" -m alternative_models.real_data_smoke inventory \
  --base "$BASE" --repo "$ALT" --out "$RUN/cohort-inventory.json"

# Recheck the mathematical smoke because the compatible timm version changed.
timeout --kill-after=30s 10m "$TRAIN_PY" \
  -m alternative_models.whole_patient_smoke \
  --backbone convnext_v2_t --device cuda:0 --tile-chunk 128 --tiles 257 \
  --out "$RUN/convnext-training-env-math.json"

timeout --kill-after=30s 30m "$TRAIN_PY" \
  -m alternative_models.real_data_smoke smoke \
  --package "$BASE/packages/AB_qc-v5" --repo "$ALT" \
  --backbone convnext_v2_t --fold 0 --tile-chunk 128 \
  --tiles-per-patient 129 --device cuda:0 --out "$RUN/convnext"
)
```

Pretrained weights may need downloading on first use. Normalization failures are
logged and raised; the pipeline does not silently substitute raw tiles. If a
partial combined output already exists from an interrupted earlier attempt, the
inventory refuses it rather than using `--allow-partial` or deleting it.
The new smoke's accuracy is diagnostic only. Preserve `receipt.json`,
`frozen_patient_metrics.json`, `finetune_patient_metrics.json` and the inventory.

## Tests and Git

The new tests use real tiny HDF5 files containing PNG bytes for both A and B,
canonical fixture labels/folds, the unchanged owner combiner/reader, identity
stain normalization and a tiny CPU encoder. They verify combination without tile
copies, both halves/classes/roles, original-pixel hash checking, corruption
rejection, one patient update in both stages, patient-level reporting and invalid
smoke receipts. They do not establish production TIAToolbox or GPU compatibility.
Run all tests under the patch environment or the full training environment.

```bash
(
set -euo pipefail
cd "$ALT"
env LD_PRELOAD="" \
  LD_LIBRARY_PATH="$BASE/alternative-models-training-env/lib/python3.12/site-packages/torch/lib" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$TRAIN_PY" -m unittest discover -s alternative_models/tests -v
git diff --check
)
```

Commit/push these four new files along with the already-applied whole-patient v2
files if those are not committed yet. Keep the records/ directory's byte-preserving
attributes intact. Use the existing GitHub CLI sign-in:

```bash
export GH_CONFIG_DIR="$BASE/github-cli-config"
(
set -euo pipefail
cd "$ALT"
GH="$(command -v gh || find "$BASE" -maxdepth 5 -type f -name gh -executable -print -quit)"
test -n "$GH"
"$GH" auth setup-git --hostname github.com
git add -- alternative_models/real_data_smoke.py alternative_models/setup_training_env.sh \
  alternative_models/tests/test_real_data_smoke.py alternative_models/REAL_DATA_SMOKE.md
git diff --cached --check
git commit --only -m "Add combined-cohort real-data ConvNeXt smoke and isolated environment" \
  -- alternative_models/real_data_smoke.py alternative_models/setup_training_env.sh \
  alternative_models/tests/test_real_data_smoke.py alternative_models/REAL_DATA_SMOKE.md
GIT_TERMINAL_PROMPT=0 git push
)
```

The four-file commit depends on whole-patient v2; commit/push the v2 files using
`WHOLE_PATIENT.md` first. No real tiles, pretrained weights, credentials or new
run outputs belong in these code commits.
