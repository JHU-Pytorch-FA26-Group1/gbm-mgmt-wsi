# Separate attention-pooling alternatives

- Scope: add code only inside `alternative_models/`. Existing `training/`, `data_prep/`, configurations and notebooks are not edited.
- Reuse: owner-maintained backbone and attention classes, verified package reader, Macenko transform, ImageNet tensor normalization, and shared flip/quarter-turn augmentation. Replacing these from scratch would add avoidable differences.
- Separate implementation: paired patient pools, quota/coverage sampler, frozen-feature cache, two-stage training, GPU benchmark and audit records.
- No job is submitted by this code. No diffusion inputs, generated images, QC decisions, real tiles, existing checkpoints, or OneDrive files are modified.

## Agreed protocol

- Train on TCGA training patients only; exclude the requested validation fold. CPTAC is excluded entirely from this CV runner, including stain fitting and feature caching.
- Require canonical patient/fold/MGMT identities and source-pixel lineage. MGMT labels are inherited patient labels; QC acceptance does not establish biological label preservation.
- For each patient, select a nested real-only pool of `min(cap, available_real_tiles)` tiles. Replace real tiles with that design's accepted variants, preserving total count. Real subsets are nested across arms and reused across backbones.
- Use every supplied accepted variant. If variants fill or exceed the budget, stop for an explicit cap/subsampling decision; do not silently drop variants or eliminate all real tiles.
- Keep patients without accepted variants: their corresponding pools remain real-only. Report this exposure; do not silently change the patient cohort.
- Actual synthetic fraction is patient/design specific. Twenty percent was an example, not a universal imposed ratio. Crop and resize may retain different numbers after QC. `patient_counts.csv` records those differences; a matched-count secondary analysis is not implemented here.
- Frozen stage: train attention pooling on every selected cached embedding in a patient's pool. The encoder is frozen and in evaluation mode; cached pixels have deterministic preprocessing. Validation features are cached with the same frozen encoder and training-only stain reference.
- Fine-tuning: update the entire backbone and attention head. Each cycle traverses every selected tile exactly once, using 256- or 64-tile bags. No replacement or padding; retain the partial last bag.
- Every bag has an integer quota determined from the cumulative target synthetic fraction. A 20% pool produces 51/52 synthetic tiles per full 256-tile bag, or 12/13 per full 64-tile bag. Cumulative discrepancy is less than one tile; a full cycle uses exactly the selected counts. Strictly 13/64 in every bag is a different fraction and does not exactly traverse a 4,000/1,000 pool without repeats or exceptions.
- Within the real and synthetic queues, spread selections across slides, spatial blocks and parent families before returning to available siblings. Independence of biological content is not assumed.
- Average bag losses by their share of the patient's tile count, accumulate gradients, then take one optimizer step per patient per cycle. This prevents more bags from automatically giving a patient more update weight. It is an average of bag-level losses, not an exact gradient of full-pool attention.
- Fixed per-fold class weight: training-patient unmethylated/methylated count. Adam, FP32, no scheduler. Learning rates, stage cycle limits and early-stopping patience are explicit required inputs, not silently chosen defaults.
- Restore the best validation-loss checkpoint at each stage. Score the same complete real-only validation pools with global attention in both stages; no chunk-wise averaging of patient predictions.
- Use identical settings, patient pools, stain reference and validation data for the three arms. Train an individual cache for each backbone/fold, shared across arms. A fine-tuned backbone cannot reuse the frozen encoder's features for validation.
- Seed all draws; request deterministic torch operations. Record effective inputs, owner/new source hashes, dependency versions, initial weights and selected checkpoints. Unsupported deterministic operations fail rather than silently switch behavior.
- These generation manifests are fold-specific. A fold-0 generation is not automatically eligible in another fold; preparation refuses the mismatch. This is not a completed five-fold augmentation study.
- Backbones currently reused: Swin V2-T and ConvNeXt V2-T. Their current pretrained recipes differ; this code does not change or claim to match those recipes.

## First: test 256 versus 64 on the intended allocation

- Use an isolated checkout based on the latest reviewed `main`; retain the generation checkout.
- The benchmark runs random tensors and random backbone weights, avoiding a pretrained download. It measures FP32 forward/backward plus Adam state, using a warmup and three measured steps. It produces no valid model-quality result.
- Run on the intended allocation when the production generation workload has released it. Do not benchmark on top of an active diffusion worker and infer isolated performance.
- Repeat for each backbone. Select one feasible bag size and use it across all arms of that backbone. Record allocated/reserved memory and leave operating headroom; a single successful tensor benchmark is not an end-to-end data throughput guarantee.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
PY="$BASE/patch-env/bin/python"
cd "$ALT"

"$PY" -m alternative_models.run benchmark --backbone swin_v2_t \
  --bags 256 64 --out "$BASE/experiments/alternative-models/swin-bag-benchmark.json"

"$PY" -m alternative_models.run benchmark --backbone convnext_v2_t \
  --bags 256 64 --out "$BASE/experiments/alternative-models/convnext-bag-benchmark.json"
```

- The intended environment needs compatible torch/torchvision, `timm` for ConvNeXt, TIAToolbox, NumPy, pandas, Pillow, h5py, PyYAML and scikit-learn. Do not blindly reinstall the owner's requirements into the running diffusion environment.
- Use the Python 3.12 patch environment for initial checks; the owner package loader uses `hashlib.file_digest` (Python 3.11+). The diffusion Python 3.10 environment is a different environment.
- Before actual runs, check imports and save the actual environment versions. Training receipts also retain versions and code hashes. No GitHub/Run:ai/rclone credentials are read into receipts.

```bash
"$PY" -c 'import torch, torchvision, tiatoolbox, timm, h5py, yaml, pandas, sklearn; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())'
"$PY" -m pip freeze > "$BASE/experiments/alternative-models/environment-freeze.txt"
```

## Prepare real and accepted synthetic pools

- Run only after the required generation/QC/review results exist. Do not use the incomplete crop job as a completed training input.
- Use a verified combined A+B package once available, or explicitly report a Half-A/Half-B-only exploratory run. No automatic data combination or extraction is performed here.
- Supply a JSON list with one entry per accepted synthetic artifact (including each original/additional shard). The image root is the actual generated shard directory to which `classifier_path` is relative. It is not the QC directory.
- Namespace ref IDs from independent Half-A/Half-B batches. Repeated QC versions of the same variants must not be included twice; duplicate IDs are refused.
- The accepted table and its complete receipt must match checksums and the canonical fold map. Synthetic image hashes are checked on decoding; real pixel hashes are checked by the shared reader. Fresh pixel-only QC is not described as fresh GrandQC inference.

```json
[
  {
    "design": "resize",
    "namespace": "half-A",
    "accepted_csv": "/absolute/path/to/current-resize-qc/accepted.csv",
    "receipt": "/absolute/path/to/current-resize-qc/receipt.json",
    "image_root": "/absolute/path/to/resize/generated/shard-0"
  }
]
```

- The example is a schema, not a claim those QC files exist. Specify actual receipts for every desired shard and design. Old original resize variants 01/02 and additional 03--08 are distinct artifacts.

```bash
"$PY" -m alternative_models.run prepare \
  --package "$BASE/packages/A_qc-v5" \
  --synthetic-inputs "$BASE/experiments/alternative-models/synthetic-inputs.json" \
  --fold 0 --cap 5000 --seed 0 \
  --out "$BASE/experiments/alternative-models/fold0-pools"
```

- Review `patient_counts.csv` for each patient's count, fraction, class coverage and missing design exposure. Inspect `source_families` in `plan.json`; QC can leave different accepted parent sets between designs.
- Omitting `--synthetic-inputs` prepares a real-only plan for a pipeline test; it does not manufacture a synthetic condition.

## Cache and train after choosing hyperparameters

- Select a reviewed reference patch ID from the current fold's TCGA training patients. The shared loader refuses a held-out or CPTAC reference. This runner does not inherit the owner's all-TCGA reference default.
- Each cache holds one common initial pretrained model/head checkpoint and union features for all planned arms plus real validation patients. Code/plan/initial-weight and feature-file changes invalidate it.
- Check available disk space before caching. A float32 768-dimensional embedding uses approximately 3,072 bytes per tile, before IDs/metadata and weights.
- Cache and output folders must be new; incomplete failed folders are preserved for inspection, not automatically deleted or resumed.

```bash
# Set these to actual reviewed inputs/settings; no placeholder is a runnable default.
: "${REFERENCE_PATCH:?Set a reviewed fold-0 TCGA training reference patch ID}"
: "${FROZEN_LR:?Choose frozen-stage learning rate}"
: "${FINETUNE_LR:?Choose fine-tuning learning rate}"
: "${FROZEN_CYCLES:?Choose frozen-stage maximum cycles}"
: "${FINETUNE_CYCLES:?Choose fine-tuning maximum cycles}"
: "${PATIENCE:?Choose validation-loss stopping patience}"
: "${BAG_SIZE:?Choose 256 or 64 from the benchmark}"

PLAN="$BASE/experiments/alternative-models/fold0-pools/plan.json"
CACHE="$BASE/experiments/alternative-models/fold0-swin-cache"

"$PY" -m alternative_models.run cache --plan "$PLAN" --backbone swin_v2_t \
  --reference-patch-id "$REFERENCE_PATCH" --chunk 16 --out "$CACHE"

for ARM in real resize crop; do
  "$PY" -m alternative_models.run train --plan "$PLAN" --cache "$CACHE" \
    --arm "$ARM" --bag-size "$BAG_SIZE" \
    --frozen-lr "$FROZEN_LR" --finetune-lr "$FINETUNE_LR" \
    --frozen-cycles "$FROZEN_CYCLES" --finetune-cycles "$FINETUNE_CYCLES" \
    --patience "$PATIENCE" --chunk 16 \
    --out "$BASE/experiments/alternative-models/fold0-swin-$ARM"
done
```

- The example loop requires all three arms in the plan. For a real-only smoke test run only `--arm real` and do not present fixture or smoke outcomes as study results.
- The same explicit cap, bag size, rates, cycle budgets and stopping rule apply across arms. Early stopping can produce different actual update counts; receipts record selected cycles. A strictly equal-duration/update sensitivity run needs a separately specified fixed-cycle protocol.
- Outputs: `bag_audit.csv`, per-cycle losses/AUROC and validation predictions, final real validation predictions, both stage-best checkpoints, and `run_record.json`. A failure leaves completion/results flags false. These flags mean the protocol finished, not that performance or biological validity is established.
- No external CPTAC scoring, automatic fold generation, matched-QC-count secondary experiment, distributed training, mixed precision, full-patient-image end-to-end attention, or automatic Run:ai submission is included.

## Tests and limits

```bash
"$PY" -m unittest discover -s alternative_models/tests -v
```

- Tests check exact 5,000-tile coverage at 64/256, quota arithmetic, diversity, reproducibility, nested equal-size pools, no overwrite, held-out/label/pixel-lineage protection, changed QC evidence, duplicate variants, cache corruption, frozen encoder behavior, fine-tuning gradients, patient update counts, failure records and best-checkpoint restoration.
- CPU integration uses a tiny backbone and deterministic fixture pixels. Actual GPU memory, pretrained downloads, real TIAToolbox data transforms, production HDF5 packages and the completed synthetic QC artifacts still need runtime validation. No such checks are claimed by the fixture tests.
