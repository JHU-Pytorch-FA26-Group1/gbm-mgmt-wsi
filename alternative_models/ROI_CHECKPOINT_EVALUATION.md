# Existing GBM checkpoint: exploratory ROI evaluation

## What is ready

The official MIE 2025 MONAI GBM segmentation project supplies a **trained binary
tumor segmentation ONNX model**, separate from its training source code. We
downloaded the actual release asset and ran CPU inference successfully. No new
model training, Virchow feature extraction, or GPU allocation is required here.

- Paper: https://doi.org/10.3233/SHTI250279
- Source: https://github.com/hnu-digihealth/monai_gbm_segmentation
- Release: `v1.0.0-MIE_Release`
- Pinned source commit: `3c3818233c3c8bcf572c1f8ad6e4181d2fa46fee`
- Released asset: `model_v3.onnx`, 26,010,685 bytes
- Observed SHA-256: `d4aef23368e1cfb3d444ecdcefc8d67376e617b19c775c4ed71fe249abfee845`

This checksum pins the bytes we downloaded; the release did not supply an
independent author-published checksum. `records/roi-checkpoint-evaluation`
contains the technical result, provenance, and upstream license.

The earlier GBM_WSSM model remains unavailable through its advertised weight
link (HTTP 404); unresolved repository issues report the same problem. Its
eight anatomical classes also do not provide an explicit normal-brain class.
The MONAI model is the more directly relevant binary candidate for this pilot.

## What has NOT been established

Technical inference success does not establish tumor-selection accuracy on A+B.
The paper reports an in-house training dataset; a patient-level overlap audit
against our development/test cohorts has not been completed. The retrieved
documentation does not specify the checkpoint's training microns/pixel or
include its stain-normalization reference. Its own reported performance is
not an estimate of performance on our cohort.

The pilot records **0.5 microns/pixel as our input condition**, not a verified
match to the checkpoint's training scale. It uses a provisional reference
selected deterministically from training tissue. Predictions are explicitly
exploratory; there is no production filtering, acceptance manifest, confusion
matrix, or Dice score without reviewed ground truth.

## Use only the existing tile packages

The packaged images are 224x224 pixels. The ONNX model expects float32
`[batch, 3, 1024, 1024]` and outputs single-channel logits.

`roi_preview.py` reconstructs each input from an **intact 5x5 neighborhood of
original tiles from the same patient and slide**, using their `x_target` and
`y_target` coordinates. This gives 1120x1120 pixels, center-cropped to 1024.
There is no resizing, missing-pixel padding, interpolation, stain-normalized
tile substitution, or joining of unrelated patches. A metadata-only geometry
scan first identifies training patients with enough intact separated regions.
Patients are then sampled reproducibly from that feasible set, without reading
pixels or using tumor predictions. Do not enlarge 224px images to bypass gaps.

Every contributing tile is checked against its recorded decoded-pixel hash,
HDF5 identity, tile size, and MPP. Package metadata and canonical fold hashes
are checked. The preview uses only TCGA patients outside the excluded outer
fold. A+B patient duplication is refused; CPTAC pixels are not read.

The target is up to four patients per half, two separated regions per patient:
**up to eight patients and sixteen region predictions**. If fewer patients are
feasible, all feasible patients are used up to that target; at least one per
half is required. `region_availability.json` records each scanned patient's
geometry counts, selection, and skip reason. Skipping concerns only this
preview and never changes classification eligibility. This is a feasibility and
qualitative review sample, not full-cohort validation. No MGMT labels or
classifier predictions are shown to the reviewer.

## Input preparation details

The authors use modified Reinhard normalization (`torchstain==1.3.0`) and
RGB values in 0..255 converted to float32. We use the same package's NumPy
backend to keep this pilot CPU-only. This is not a claim of bitwise identity
to the authors' Torch backend. The reference and input hashes are recorded.
The downstream MGMT model's existing Macenko configuration is unchanged.

MONAI 1.3.2's default `PILReader` reverses the two spatial axes. The adapter
therefore creates CWH inputs and reverses the output axes before drawing
overlays. The alignment test verifies this with asymmetric image content.
The released one-channel logits receive a sigmoid; the displayed binary mask
uses probability >0.5, matching the upstream inference rule. The percentage
is **predicted tumor area**, not validated tumor purity or calibrated certainty.

## Run on the existing Run:AI workspace

Unzip `gbm-roi-checkpoint-evaluation.zip` to
`/workspace/persist/crivas4-microscopy/gbm-roi-checkpoint-evaluation`.
The archive includes the tested ONNX asset; there is no need to download the
broken GBM_WSSM weight link or install TensorFlow.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
BUNDLE="$BASE/gbm-roi-checkpoint-evaluation"

(
set -euo pipefail
cd "$ALT"

if git apply --check "$BUNDLE/roi-evaluation.patch"; then
  git apply "$BUNDLE/roi-evaluation.patch"
elif git apply --reverse --check "$BUNDLE/roi-evaluation.patch"; then
  echo 'ROI patch already applied.'
else
  echo 'Patch does not match this checkout; stopped.'
  exit 1
fi

bash alternative_models/setup_roi_env.sh
ROI_PY="$BASE/alternative-models-roi-env/bin/python"
OUT="$BASE/experiments/alternative-models/roi-preview-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$ROI_PY" -m unittest discover \
  -s alternative_models/tests -p test_roi_preview.py -v

"$ROI_PY" -m alternative_models.roi_checkpoint \
  --checkpoint "$BUNDLE/checkpoint/model_v3.onnx" \
  --out "$OUT/technical-probe.json"

"$ROI_PY" -m alternative_models.roi_preview \
  --base "$BASE" --repo "$ALT" \
  --checkpoint "$BUNDLE/checkpoint/model_v3.onnx" \
  --exclude-fold 0 --patients-per-half 4 --regions-per-patient 2 \
  --out "$OUT/review"

printf 'Review folder: %s\n' "$OUT/review"
)
```

The new `alternative-models-roi-env` is isolated and CPU-only. It does not
change the generation/classification environments or submit a Run:AI job.
The default process uses two CPU threads. Complete scans of the package CSVs
are required, but image reads are limited to the selected original regions.

The coordinate scan streams one slide at a time and keeps only candidate
regions in memory. If an entire half has no training patient with the requested
number of separated regions, the command writes `<out>-availability.json`
beside the requested output and stops without publishing region images.
The successful review folder also contains this geometry report. Report a
failure before changing sample size, seed, or reconstruction rules.

## Pathologist review

Download the entire `review` folder, then open `review.html`; the image assets
must remain beside the HTML. Each region shows original RGB, the normalized
model input, the predicted tumor overlay, and the binary prediction.

In `review.csv`, record `usable`, `needs_correction`, or `unusable` and comments.
Specifically inspect **missed infiltrating/cellular tumor** and **non-tumor
tissue falsely marked as tumor**. Retain the exported masks for targeted
correction if useful. These initial judgments are feasibility feedback, not
an independent quantitative validation set.

Before automatic whole-cohort selection, resolve physical-scale compatibility,
assess the normalization choice, and validate retained/rejected tissue on a
patient-disjoint reference sample. Any filter/threshold selected using our
annotations must remain inside the relevant training fold. Both downstream
backbones must receive the same frozen selection policy.

## Publish only the new alternative implementation

After successful tests, the following commits the new ROI code/docs/receipts
without changing existing teammates' model files or committing model weights:

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
export GH_CONFIG_DIR="$BASE/github-cli-config"
(
set -euo pipefail
cd "$ALT"
GH="$(command -v gh || find "$BASE" -maxdepth 5 -type f -name gh -executable -print -quit)"
test -n "$GH"
"$GH" auth setup-git --hostname github.com
FILES=(
  alternative_models/roi_checkpoint.py
  alternative_models/roi_preview.py
  alternative_models/setup_roi_env.sh
  alternative_models/ROI_CHECKPOINT_EVALUATION.md
  alternative_models/tests/test_roi_preview.py
  alternative_models/records/roi-checkpoint-evaluation
)
git add -- "${FILES[@]}"
git --no-pager diff --cached --check -- "${FILES[@]}"
if ! git diff --cached --quiet -- "${FILES[@]}"; then
  git commit --only -m 'Add exploratory evaluation of a released GBM tumor segmentation checkpoint' -- "${FILES[@]}"
fi
GIT_TERMINAL_PROMPT=0 git push
)
```

No accuracy improvement or tumor filtering has been claimed or performed by
this delivery. The next evidence is the A+B preview and the pathologist's review.
