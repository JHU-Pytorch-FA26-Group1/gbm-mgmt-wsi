# Corrected matched-source resize target: 16,000 raw variants

- The target is **16,000 generated variants for EACH design**, before QC: 32,000 across crop and resize arms.
- The archive audit found only **4,000 unique original resize variants**, not 8,000. The two 2,000-output shards, older QC archives and combined v4 archive contain the same variants. A native output and its enlarged classifier copy are one variant.
- The matched source pool is the original **2,000 TCGA training tiles** (1,000 from each source-patient MGMT class), covering the same 63 eligible patients in held-out fold 0 / Half A. The active crop arm uses exactly these originals.
- Crops: 2,000 originals × four 128-pixel corner crops × two variants = 16,000. Resize: 2,000 originals × eight variants = 16,000. Resize variants 01/02 already exist; this job generates **03--08**, totaling 12,000 additional outputs.
- No source selection, new tiles, patient reshuffling, class-conditioning change or fold change is permitted. Preparation copies the original prepared references byte-for-byte, checks the crop-source lineage against them, and refuses any difference.
- The earlier `fold-0-resize-newtiles-4000` run selected disjoint new tiles and is outside this matched comparison. Its workload was deleted at Carlos's request to release compute; its outputs and records remain on the existing PVC. This replacement does not read or delete those outputs.

## 1. Source and budget audit

- Preparation verifies original shard completions, variant identities, seeds, generation-manifest hashes and source/output image hashes. It requires original variants 01/02 and matching crop patient/slide/tile/fold/label/pixel lineage.
- Original references, fold and label files and allocation CSV are copied into the new batch. No package resampling or new TileReader draw occurs. Source hashes, original completions and crop input metadata are anchored in the new plan.
- The original seed namespace is extended using the same `legacy.seed_for(original_seed, reference_id, 'variant', index)` for indices 3--8. The original seed remains 42 for the established run. All additional seeds must be unique and distinct from original seeds.
- Variant IDs are `<original reference ID>-03` through `-08`. They cannot overwrite `-01`/`-02`, and all new files live in a separate output directory.

```bash
BASE=/workspace/persist/crivas4-microscopy
cd "$BASE/gbm-mgmt-wsi"
bash scripts/run_hismmdm_resize_matched.sh prepare
bash scripts/run_hismmdm_resize_matched.sh validate
```

## 2. Generation procedure

- Reuse the established Python 3.10 His-MMDM environment and verified HMU-1st checkpoint. Keep upstream commit `26a1f9f12a8a3d1799496fb7dacebd722d92f3e3`, H&E class 10 in both directions, frozen FP16 weights, 1,000 DDIM steps, eta-zero inversion without clipping, eta-0.3 decoding with clipping, and disabled classifier guidance.
- This extension reuses `core.make_tensor`, `core.decode`, `core.to_image`, `core.release_settings` and the legacy seed function. It does not reimplement sampler mathematics or alter the existing core/batch scripts.
- Preserve the original resize execution batch size of **one image**, including deterministic algorithms and disabled cuDNN benchmarking. Its runtime/GPU type must match the original resize receipts. The crop arm's existing batch-size-8 execution is preserved and remains separately recorded; bitwise pairing with crop outputs is not claimed.
- Each original is resized from 224 to 128 using the same upstream full-field preprocessing as before. Every prepared 128-pixel generator input must exactly match its original saved native input.
- A newly recomputed eta-zero reconstruction must exactly match the original reconstruction pixels for every source. On the inherited verification subset, also reproduce original variant 01 with its recorded seed, check eta-zero seed independence, and repeat new variant 03 with the same seed. Any discrepancy stops generation rather than silently changing the method.
- Generate six new stochastic variants per original. Reject duplicate output pixels within that source across old/new variants, nonfinite model output, source-lineage mismatch or incompatible resume metadata. These are technical checks, not biology/QC acceptance.
- Native variants remain 128 × 128; separately saved classifier copies are 224 × 224 bicubic, retaining the original full 112-µm field of view. The crop arm retains its intentionally smaller 64-µm field of view. MGMT is inherited source-patient metadata, not an input conditioning label or proof of synthetic molecular status.

## 3. Submit on the freed half GPU

```bash
bash scripts/run_hismmdm_resize_matched.sh submit
runai training standard list --project runai-users
```

- Request 0.5 GPU on one device, four CPU cores and 16 GB CPU RAM, with the confirmed image and existing PVC. Omit the optional fractional GPU limit, which this cluster rejected without GPU Optimization. Use ordinary preemptible training priority.
- The new workload name starts `hismmdm-resize-matched12k-`. It runs both 6,000-output additional shards sequentially in the foreground, verifies the combined 16,000-output ledger, then exits. It never launches another source draw or changes the crop workload.
- Exact code snapshots, hashes, Git/CLI versions and runtime logs are stored in `fold-0-resize-matched-additional12000/runai/<job>/`. Authentication configuration is never copied.
- A 94-hour running-time limit plus a two-minute forced-termination grace bounds each attempt. Automatic deletion is ten minutes after completion. The existing PVC and saved outputs survive workload deletion. Partial source work can be recomputed; completed source receipts are checked and skipped on resumption.
- **Time planning:** six new variants reuse one inversion and reconstruction per original. There are 16,000 normal single-image diffusion passes plus inherited verification overhead. At the previously observed half-GPU rate of roughly 21 seconds/pass, generation is approximately 94--95 hours. This is not a new measurement or a guarantee. The 94-hour cap or preemption may require resubmitting the same job for the unfinished tail. Do not promise completion within four days or silently increase resources/change batching to obtain it.
- On resubmission, reuse the prepared batch and unchanged code/runtime; run only `submit`, not `prepare`. A global worker lock protects against duplicate active jobs. A different execution signature stops before overwriting completed outputs.

## 4. Progress and final count

```bash
watch -n 60 bash scripts/run_hismmdm_resize_matched.sh status
bash scripts/run_hismmdm_resize_matched.sh verify
```

- Status prints the retained 4,000, additional completed variants, the source-matched total out of 16,000, and an ETA including BOTH additional shards. It reads files without a Run:ai login. The old monitor's `resize-newtiles` row refers to the deleted unmatched run and is not the corrected resize count.
- Final verification requires exactly variants 01--08 for every original tile, eight distinct native pixel hashes per tile and 8,000 raw outputs per source-patient MGMT class. It writes `matched_resize_16000.csv` and a combined completion receipt.
- The combined CSV records `artifact_batch` and `artifact_shard` alongside each image's existing relative path. Original files stay in the original folder; additional files stay in the new folder. Do not flatten, overwrite, or count classifier copies as extra variants.
- Output folder: `GBM_MGMT/hismmdm-pilot/fold-0-resize-matched-additional12000`. No automatic upload, QC acceptance or training promotion occurs. Apply the same agreed QC procedure to new outputs and audit retained per-patient coverage afterward. Existing accepted/rejected decisions on the original 4,000 are preserved separately.

## 5. Code and tests

- Commit this document, `his_mmdm_resize_matched.py`, the launcher and tests before preparation/submission. Original generation, wrong-source extension and crop scripts are left unchanged for provenance.
- CPU fixtures exercise original-source/crop matching, budgets, appended seed/variant identities, original-control and variant replay gates, six-output source receipts, resumption and corruption detection. Sampling is mocked in CPU orchestration tests; actual GPU/model replay checks execute on the cluster before new outputs for verification sources are committed.

```bash
# The existing inference environment already contains Torch/Pillow/skimage.
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$BASE/hismmdm-env/bin/python" \
  -m unittest discover -s tests -p test_hismmdm_resize_matched.py -v
bash -n scripts/run_hismmdm_resize_matched.sh
```
