# MGMT microscopy reproducibility record · November 25, 2026

## Scope and evidence

This record covers the decisions and terminal evidence in this conversation and the recovered November 20–25 handoff from “Audit IDH WT Cohort.” The preceding October conversation was unavailable in full. The original slide inventory comprised 524 distinct SVS paths for all 234 canonical patients (320 slides/151 TCGA development patients and 204 slides/83 CPTAC external patients). A and B partition data preparation ownership. TCGA supplies five patient-level folds. CPTAC cannot select thresholds, epochs or model hyperparameters. Earlier QC calibration involved 26 TCGA and 12 CPTAC patients and should be disclosed separately.

The recovery branch starts from feat/alternative-models at 96bd0e46cc68523557bd6fc20690699b9678e5d9. It includes the recovered whole-patient implementation, real-data smoke tooling, repaired ROI preview, two read-only audits and the direct-WSI launcher. The older classifier audit needs compatibility review. See WHOLE_PATIENT.md, REAL_DATA_SMOKE.md, ROI_CHECKPOINT_EVALUATION.md and HALF_B_GENERATION.md here for component-specific commands. No owner training/ files are changed.

## Why the extraction design changed

The successful CPU feasibility preview used the MIE 2025 MONAI GBM binary U-Net (DOI 10.3233/SHTI250279), released model_v3.onnx, SHA-256 d4aef23368e1cfb3d444ecdcefc8d67376e617b19c775c4ed71fe249abfee845. With only packaged tiles available, the repaired sampler sought complete 5×5 neighborhoods of adjacent real 224-pixel tiles, cropped 1120 to 1024 and selected two separated regions per patient. Sixteen regions were reviewed technically across eight fold-0 TCGA training patients. All 64 panel paths resolved and the saved masks and overlays matched the probabilities. Regions 005 and 006 each had one positive bottom-right pixel. Region 008 was entirely positive. Expert judgments remain pending.

The metadata audit showed that 52 of 118 fold-0 training patients with model tiles had zero intact windows. Two separated regions were available for 35/63 in A and 28/55 in B. Even processing every intact window would fully cover only 150,933 of 1,130,554 training tiles. Once original WSIs were transferred to Run:AI, the team chose direct WSI reads and retained the frozen A+B classifier pool as the scoring target.

| Half | Frozen classifier tiles | Occupied 896-pixel blocks |
|---|---:|---:|
| A | 989,810 | 163,321 |
| B | 997,407 | 160,441 |
| Combined | 1,987,217 | 323,762 |

The pool spans 521 slides. Group by slide and floor-divide target-resolution x and y by 896. Each occupied block gets a 1024×1024 read at 0.5 µm/pixel with 64 pixels of neighboring context on each side. The reader shifts edge windows inward. It maps a predicted mask back onto each 224×224 footprint, writing a fraction or an insufficient-context reason. The frozen real GrandQC decisions are inherited through the package manifests. Segmentation preprocessing uses torchstain modified Reinhard and the saved region-001 reference. Downstream classifier normalization uses Macenko. ROI model training MPP, original stain reference, training-cohort overlap and local anatomical accuracy remain unverified. The 0.5 pixel probability cutoff produces a mask; a classifier tile retention threshold is still undecided.

## Run:AI commands

Use the existing workspace. Paths below reproduce the observed deployment. On a fresh checkout obtain the exact checkpoint and pilot reference first; the preview setup and tests are documented in ROI_CHECKPOINT_EVALUATION.md. Reuse the original region-001 reference and receipt when reproducing the November run. Keep credentials, slide data, GPU wheels and patient output CSVs outside Git.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
EXP="$BASE/experiments/alternative-models"
RUN="$EXP/roi-preview-fixed-20261125-034609"
OUT="$EXP/direct-wsi-roi-v1"
GPU_PY="$BASE/alternative-models-gpu-env/bin/python"
ORT_DIR="$EXP/ort-gpu-1.22.0"
LAUNCH="$ALT/scripts/run_hismmdm_half_b.sh"
BROOT="$BASE/GBM_MGMT/hismmdm-pilot/half-B-fold-0-matched-32000"
export PATH="$BASE/bin:/root/.runai/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"
export RCLONE_CONFIG="$BASE/rclone-config/rclone.conf"
cd "$BASE"
```

Original source was gdrive:GBM_MGMT. The completed transfer used an SVS-only rclone copy. These inventory commands can be repeated independently.

```bash
rclone size gdrive:GBM_MGMT --include '*.svs' --ignore-case
find "$BASE/GBM_MGMT/raw" -type f -iname '*.svs' | wc -l
# For a fresh destination only:
# rclone copy gdrive:GBM_MGMT "$BASE/GBM_MGMT" --include '*.svs' --ignore-case
```

Review artifacts and metadata-only coverage audit (choose an unused output name when repeating):

```bash
(cd "$RUN" && python3 -m zipfile -c roi-review-export.zip review review-availability.json)
python3 "$ALT/alternative_models/audits/roi_coverage_audit.py" \
  --base "$BASE" --repo "$ALT" --receipt "$RUN/review/receipt.json" \
  --out "$RUN/roi-tile-coverage-repeat.csv"
```

GPU execution was verified with Torch 2.8.0+cu128, CUDA 12.8 and onnxruntime-gpu 1.22.0 on the 0.25 allocation of an RTX PRO 6000 Blackwell Server Edition. A profiled zero-input forward pass recorded Conv and ConvTranspose on CUDA. The existing ROI environment uses CPU ONNX Runtime. For a fresh GPU wheel target the install command was:

```bash
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT "$GPU_PY" -m pip install \
  --no-deps --only-binary=:all: --target "$ORT_DIR" onnxruntime-gpu==1.22.0
```

The deployed launcher lives at $BASE/run_wsi_roi.py; the identical source is archived in alternative_models/run_wsi_roi.py. It imports ROI helpers from the repaired ALT checkout and starts slide readers in patch-env. Compare sources before reproducing from a different checkout. Launch once, then use the read-only monitoring commands:

```bash
sha256sum "$BASE/run_wsi_roi.py" "$ALT/alternative_models/run_wsi_roi.py"
mkdir -p "$OUT"
nohup env PYTHONPATH="$ORT_DIR" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$GPU_PY" -u "$BASE/run_wsi_roi.py" >> "$OUT/run.log" 2>&1 </dev/null &
tail -n 20 "$OUT/run.log"
find "$OUT" -maxdepth 1 -name '*.csv' | wc -l
```

The launcher writes per-slide CSVs atomically after finishing a slide. A restart checks existing patch IDs and redoes a partial slide. Reuse the output directory only for identical settings and inputs. Full-run CSVs contain tumor fractions, identifiers, coordinates and reasons; they do not store full masks. The last supplied log showed 271,000/323,762 blocks at 5.86/s, with 2.5 h projected. The agreed ceiling was 24 h (3.75 blocks/s required). Completion, unscored count and final wall time await the COMPLETE line and output reconciliation. No threshold-based training eligibility change has been applied.

## Matched diffusion variants

Both designs use the same 2,000 original fold-0 training tiles per half, 1,000 per class. Resize generates eight variants per original. Crop uses four 128-pixel corners and two variants per corner. A prior extension using different original sources was rejected and excluded. A resize combines 4,000 unique original variants and a matched extension of 12,000. Preserve source pixel hashes, parent patient, generator settings, native outputs and classifier-copy lineage. Never count a resized classifier copy as another variant.

| Arm | Raw target | Latest observed evidence |
|---|---:|---|
| A crop | 16,000 | Both shards finalized |
| A resize | 16,000 | Final completion gate passed before B resize submission |
| B crop | 16,000 | Running on 1 GPU in last live scheduler table |
| B resize | 16,000 | hismmdm-halfb-resize-20261125-221003 Running, 1/1 pod, 0.50 GPU at 22:10 UTC |

```bash
# If the scheduler token expires:
runai login sso
runai training standard list --project runai-users
bash "$LAUNCH" a-status
bash "$LAUNCH" status
find "$BROOT/runai" -maxdepth 2 -name job-name.txt -print -exec cat {} \;
runai training standard describe hismmdm-halfb-resize-20261125-221003 -p runai-users
tail -n 30 "$BROOT/runai/hismmdm-halfb-resize-20261125-221003/runtime.log"
# Historical dispatch (already submitted; do not repeat for this run):
# bash "$LAUNCH" submit-crop
# bash "$LAUNCH" submit-resize
```

A completion and release checks gate the corresponding B submission. The B workers run frozen blur-10 synthetic QC after generation. Jobs are preemptible, have a 94-hour worker timeout and auto-delete ten minutes after completion. Confirm persistent generation and QC receipts for each arm. Reconcile 64,000 unique raw variants by half/design/parent/variant and report accepted counts separately. OneDrive transfer and destination verification are pending. Share manifests, parent/fold assignments and QC decisions with teammates. Current variants are specific to fold-0 training; verify parent eligibility before use in any other fold.

## Downstream plan

Pathologist reviews original tissue and predicted overlays for missed infiltrating tumor and falsely marked normal tissue, blood or necrosis. Tune ROI decisions using appropriate training partitions. Primary alternative is ConvNeXt V2 Tiny first, then Swin V2 Tiny, with one global attention normalization, one weighted BCE and one prediction per patient. Use controlled final-stage fine-tuning, plateau stopping, at most 200 epochs, confusion matrices and patient-level metrics. The existing development runner uses validation for checkpoint selection. Independent outer scoring and final external evaluation remain pending. Freeze decisions on TCGA before final CPTAC evaluation.
