# Chunked whole-patient attention: ConvNeXt first

This changes only `alternative_models/`. Existing backbone, attention, data
preparation, QC and diffusion implementations remain owned and maintained by
their authors. No job is submitted and no package/image/QC record is modified.

## Objective and coverage

One patient produces one globally normalized attention representation, one logit,
one weighted BCE and one optimizer update. A 128-tile chunk is an image-processing
unit, not a separately supervised bag. The partial final chunk is retained.

Fine-tuning uses the chain rule in two encoder passes:

1. Encode each chunk without retaining image activation graphs. Keep embeddings
   on CPU, together with the encoder's torch CPU/CUDA RNG states.
2. Concatenate embeddings; calculate the single patient loss with the unchanged
   attention head. Differentiate with respect to head parameters and embeddings.
3. Recreate each chunk and its encoder RNG state. Backpropagate its embedding
   gradients through the encoder. Compare replayed features before accepting them.
4. Update trainable parameters only after the complete patient has contributed.

This preserves the global objective, rather than averaging independently
normalized chunk predictions or losses. Numerical agreement is tested to
explicit floating-point tolerances. Forward passes with mutable encoder buffers
are refused; training-mode BatchNorm is unsupported. The two current backbones
use LayerNorm. Image graphs are bounded by chunk size; CPU embeddings/gradients
and the global attention graph still grow with the patient's tile count. This
trades extra encoding and image I/O for lower image activation memory.

Every tile in the declared pool is visited exactly once per epoch, with a second
encoder pass for its gradient. Existing slide/spatial-block/family ordering and
exact cumulative synthetic quotas remain processing-order rules. No distance
filter or new sampling exclusion is introduced. Geometric separation would be a
separate selection policy if a future experiment subsamples.

`prepare --all-tiles` gives the real-only arm every available model tile per
patient, with no 5,000-tile cap. Synthetic arms preserve the agreed equal total
budget by replacing real tiles with all supplied accepted variants. Thus full
coverage means full coverage of each arm's recorded pool; an augmentation arm
does not additionally retain the real tiles it replaced. `--cap N` remains an
explicit bounded-pool experiment. No synthetic source or design is regenerated.

## Controls, stopping and reports

- Frozen stage: backbone fixed, global attention over the complete cached pool.
  Initial head learning rate: `1e-4`.
- Fine-tuning: unfreeze the final encoder stage and final normalization layers.
  ConvNeXt V2 uses `stages[-1]`, `norm_pre` and `head` (the owner factory has
  already removed its classifier). Swin V2 uses `features[-1]` and `norm`.
  Earlier encoder parameters remain frozen. Head rate: `1e-5`; encoder: `1e-6`.
- Adam, FP32 and the existing training-patient class weight; no scheduler.
- At most 200 epochs total: defaults are 20 frozen + 180 fine-tuning. Each stage
  stops after 10 consecutive epochs without an improvement of at least `1e-4`
  relative to its last significant improvement in development loss. Worsening
  also counts toward patience. These are starting settings, not tuned optima.
- Frozen plateau advances to fine-tuning; fine-tuning plateau ends the run.
  Save every actual minimum, independently of the patience tolerance. Retain the
  better stage checkpoint, so unsuccessful fine-tuning cannot replace the better
  frozen model. `run_record.json` records epochs, stop reasons and selected stage.
- `patient_metrics.json` and `confusion_matrix.csv`: accuracy, balanced accuracy,
  TN/FP/FN/TP, precision, sensitivity, specificity, F1, AUROC and the accuracy of
  the training-majority-class predictor. Rows are true classes and columns are
  predicted classes, both ordered unmethylated then methylated.
- `predictions.csv`: exactly one row per development patient, including score,
  predicted label, correctness and threshold. Current threshold is fixed at 0.5;
  no development-selected threshold or patient bootstrap is claimed here.
- `chunk_audit.csv`: patient/epoch/chunk tile IDs and real/synthetic counts.
  Chunks are not predictions or independent classification examples.

The current `train` command is a **development run**: its held-out patients select
checkpoints. Its receipts explicitly mark that this is not an unbiased outer-CV
test estimate. Inner selection, fixed-budget refitting on all outer-training
patients, independent outer scoring and final TCGA-to-CPTAC evaluation still need
their own reviewed orchestration. CPTAC is not encoded or scored by this runner.
The all-TCGA final fit must inherit a stopping budget from development, rather
than using CPTAC loss for early stopping. The current fold-0 synthetic receipts
also remain fold-specific. None of this patch claims a finished five-fold study.

## Apply and CPU regression tests

The patch is based on `feat/alternative-models` commit `96bd0e4`. Apply in the
existing alternative worktree, leaving the generation checkout alone. Download
`alternative-models-whole-patient-v2.patch` into the persistent base folder first.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
PATCH="$BASE/alternative-models-whole-patient-v2.patch"

(
set -euo pipefail
cd "$ALT"
git apply --check "$PATCH"
git apply "$PATCH"

env LD_PRELOAD="" \
  LD_LIBRARY_PATH="$BASE/patch-env/lib/python3.12/site-packages/torch/lib" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$BASE/patch-env/bin/python" \
  -m unittest discover -s alternative_models/tests -v
git diff --check
)
```

Tests include global-loss/gradient/Adam-update equivalence, uneven chunk sizes,
dropout RNG replay, unchanged frozen layers, chunk-order invariance without
dropout, replay mismatch rejection, complete coverage, one update per patient,
confusion orientation, plateau semantics, the 200-epoch cap, best-stage selection,
combined A+B gates and the existing generation/transfer tests.

## First quarter-GPU smoke: no training-data dependencies needed

This uses the existing CUDA benchmark environment and owner architecture factory,
but **random weights and random images**. It downloads no pretrained weights and
does not import HDF5, pandas, scikit-learn or TIAToolbox. Run inside your existing
quarter-GPU workspace; it submits no new Run:ai workload and touches no diffusion
process. Do not run it concurrently with another GPU task in the same workspace.

It first compares a five-image direct global loss and parameter gradients against
the chunked implementation. It then performs one patient update on 257 synthetic
images: 128 + 128 + 1, including Adam-state allocation and training-mode encoder
RNG replay. It records peak CUDA memory. This is a bounded implementation/memory
test, not a classification result or a real-data throughput benchmark.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
GPU_PY="$BASE/alternative-models-gpu-env/bin/python"

(
set -euo pipefail
cd "$ALT"
unset PYTHONPATH PYTHONHOME
export LD_PRELOAD=""
export LD_LIBRARY_PATH="$BASE/alternative-models-gpu-env/lib/python3.12/site-packages/torch/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

timeout --kill-after=30s 10m "$GPU_PY" \
  -m alternative_models.whole_patient_smoke \
  --backbone convnext_v2_t --device cuda:0 \
  --reference-tiles 5 --tile-chunk 128 --tiles 257 \
  --out "$BASE/experiments/alternative-models/convnext-whole-patient-smoke.json"
)
```

The output path must be new; failed receipts are preserved. A failed smoke exits
nonzero, with `complete: false`. Never interpret either success or failure as
evidence about MGMT prediction. Swin uses the same command with `swin_v2_t` and a
distinct output filename, after ConvNeXt. Full real-data execution remains gated
on the isolated data dependencies, combined input and a separate real-data smoke.

## Combine A+B without copying tiles

After the existing dependency review is resolved, use the owner-maintained
combiner. It checks policy/fold compatibility and duplicate patient ownership;
the output is a new index referencing the two original packages. Do not use
`--allow-partial` or bypass incompatibilities for production classification.

```bash
cd "$ALT"
# Use an environment where the verified owner loader dependencies are installed.
"$BASE/patch-env/bin/python" data_prep/combine.py \
  --parts "A=$BASE/packages/A_qc-v5" "B=$BASE/packages/B_qc-v5" \
  --out "$BASE/packages/AB_qc-v5" \
  --folds "$ALT/manifests/patient_folds.csv"
```

Before caching, inspect the combined counts and documented patients with no model
tiles. A full-cohort plan uses:

```bash
"$GPU_PY" -m alternative_models.run prepare \
  --package "$BASE/packages/AB_qc-v5" --fold 0 --all-tiles --seed 0 \
  --out "$BASE/experiments/alternative-models/fold0-real-whole-patient-plan"
```

That command needs the complete isolated data environment; it is not the next
GPU smoke command. Full-cohort preparation refuses a single-half root, incomplete
or aliased A/B inputs, duplicate canonical patients and duplicate tile IDs.
The combined index and part manifests/modeling tables are hash-recorded.
`--allow-single-package-smoke` is a deliberate invalid-for-results fixture escape
hatch, never a way to report half-only training as full-cohort training.

Existing frozen caches are source-hash-bound and become stale after this change.
Create new plan/cache/run directories; this patch does not delete or migrate old
ones. Both current pretrained recipes remain unchanged and differ between the
backbones; record that limitation when comparing complete pretrained systems.

## Commit and push after the tests

```bash
export GH_CONFIG_DIR="$BASE/github-cli-config"
(
set -euo pipefail
cd "$ALT"
GH="$(command -v gh || find "$BASE" -maxdepth 5 -type f -name gh -executable -print -quit)"
test -n "$GH"
"$GH" auth setup-git --hostname github.com

FILES=(
  alternative_models/whole_patient.py
  alternative_models/whole_patient_smoke.py
  alternative_models/patient_metrics.py
  alternative_models/engine.py
  alternative_models/prepare.py
  alternative_models/run.py
  alternative_models/sampling.py
  alternative_models/tests/test_whole_patient.py
  alternative_models/tests/test_engine.py
  alternative_models/tests/test_prepare.py
  alternative_models/README.md
  alternative_models/WHOLE_PATIENT.md
)
git add -- "${FILES[@]}"
git diff --cached --check -- "${FILES[@]}"
git commit --only -m "Use chunked whole-patient attention with exact gradient replay" -- "${FILES[@]}"
GIT_TERMINAL_PROMPT=0 git push
)
```

This uses the existing GitHub CLI sign-in, not a GitHub account password.
