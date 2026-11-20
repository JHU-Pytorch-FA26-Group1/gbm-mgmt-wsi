# His-MMDM four-corner arm and full-GPU execution

- This is a separate, matched augmentation arm. The completed 4,000 resize-arm variants and their QC files remain intact.
- The exact existing 2,000 originals are reused: 1,000 from each patient MGMT status, spread across the same training patients. The source-patient label is provenance, not a molecular conditioning input or proof of synthetic MGMT biology.
- Each 224-pixel original yields four 128-pixel crops at `(0,0)`, `(96,0)`, `(0,96)`, `(96,96)`. Adjacent crops overlap by 32 pixels along each axis. Cropping preserves 0.5 µm/px; each crop covers 64 × 64 µm, versus 112 × 112 µm in the full-field resize arm.
- Two variants per crop produce 16,000 planned outputs, with 8,000 in each inherited shard. No original or generated image is deleted.
- The existing frozen HMU-1st checkpoint, upstream commit, 1,000 DDIM steps, H&E class 10 in both directions, inversion eta 0, generation eta 0.3, FP16 model, and disabled classifier guidance are retained.
- Native generated outputs remain 128 × 128. A separately named 224 × 224 bicubic copy is saved for classifier compatibility; it has the crop's 64-µm field of view (effective spacing 64/224 ≈ 0.286 µm/px) and does not restore missing tissue context or detail. Never interpret it as a full 112-µm tile or paste the four stochastic images into a slide. Comparing this arm with the resize arm changes field of view and apparent magnification as well as input resampling.
- Real-source QC does not certify every corner crop. All four crops are retained for this matched arm; output QC and pathologist review remain pending, with no automatic training eligibility or silent approval of blank corners.

## 1. Environment and preparation

- Use the existing Python 3.10 His-MMDM environment and verified HMU-1st weights. No environment replacement, new generator training, or dependency upgrade is required.
- The shared dependency is `experiments/hismmdm-pilot/his_mmdm_he_augment.py`, already present in the original batch bundle and used on Run:ai. The blur-10 branch does not track it, while current main tracks an earlier version. Preserve the file actually used on Run:ai and add it explicitly to this feature commit; do not overwrite it with a different version during a pull. The new patch intentionally leaves it unchanged. Both the current-main helper and the batch-bundle helper were checked against the new CPU tests.
- The confirmed runtime is Python 3.10.22, Torch 2.8.0+cu128, CUDA 12.8, on the NVIDIA RTX PRO 6000 Blackwell Server Edition. The full-GPU check exposed 94.97 GiB and passed CUDA and persistent-path checks.
- The confirmed cluster/project are `runai-cluster1` / `runai-users`; image `nvcr.io/nvidia/pytorch:25.06-py3`; existing PVC `persistant-datasource-runai-cluster1-f06e9` mounted at `/workspace/persist`.
- The launcher records the CLI version and Git revisions, snapshots the exact executing Python/core/shell code, and hashes the snapshots before submission. It never records authentication tokens or configuration files. The image tag is recorded; an immutable image digest has not been supplied.
- Parent plan, reference manifest, original file/pixel hashes, labels, and fold assignments are checked. Only the parent batch's TCGA training patients are allowed. No new patient selection or fold reshuffling occurs.

```bash
BASE=/workspace/persist/crivas4-microscopy
cd "$BASE/gbm-mgmt-wsi"
bash scripts/run_hismmdm_corners.sh prepare
```

- Preparation creates a new folder, `GBM_MGMT/hismmdm-pilot/fold-0-corners-16000`. It refuses to overwrite an existing prepared folder. Reuse an existing valid preparation instead of rerunning this step.

## 2. Benchmark before full generation

- The benchmark requests one GPU device with both requested and maximum GPU portion set to 1. The worker additionally refuses this particular cluster's fractional GPU view if fewer than 90 GiB are visible.
- The first job runs two batches at size 4 and two batches at size 8, then exits. Each size has its own output directory; benchmark images are not counted as production images.
- It uses real inversion, reconstruction, two stochastic decodes, and technical repeats on the inherited repeat-check inputs. It does not reduce the diffusion step count or change eta.
- The same-seed stochastic check must reproduce exactly, eta-zero reconstruction must be seed-independent, outputs must be finite, and different variant seeds must produce different pixels. These are technical checks, not pathology validation.
- The first batch is excluded from timing extrapolation when a second batch is available. The estimate accounts for production reconstruction and inherited repeat-check overhead; it excludes scheduling, preemption, startup, later QC/review, and upload.

```bash
bash scripts/run_hismmdm_corners.sh submit-benchmark
runai training standard list --project runai-users
```

- Read `runtime.log` under the job's printed persistent-record directory. Benchmark summaries are also saved under `fold-0-corners-16000/benchmarks/<job>/bs4/generated/shard-0/completed.json` and the corresponding `bs8` path.
- If a larger batch needs testing, `bash scripts/run_hismmdm_corners.sh submit-benchmark 16` benchmarks only size 16 with separate results. Production size 16 is permitted only after that matching benchmark passes.
- The benchmark has a two-hour runtime limit. Logs and results remain on the existing PVC after job deletion.

## 3. Batched RNG and reproducibility

- The model receives only H&E class IDs. MGMT status never enters a model tensor.
- Batch membership and row order are fixed. Each batch/variant gets a deterministic recorded seed derived from the crop IDs and parent seed. Rows use distinct positions within the batch's Gaussian stream. `seed`, `rng_scope`, `rng_row`, and `rng_batch` record this honestly; there is no claim of an independent manually seeded stream per image.
- Batching changes RNG assignment and may change floating-point results compared with single-image inference. Changing the batch size creates a different execution signature; resuming an existing production folder with a different size is refused. No cross-batch-size pixel identity is claimed.
- Checkpoint, upstream revision, runner/core hashes, crop-plan hash, sampler flags, runtime, GPU type, and batch size enter the execution signature. GPU UUID and job name do not, so a matching replacement device/job can resume after preemption.
- Completion receipts are written atomically per batch only after output hashes and technical checks are available. An interrupted unfinished batch is recomputed; completed batches are verified and skipped. Corrupted or incompatible receipts stop the run rather than being silently accepted.

## 4. Submit the production job after assessing the benchmark

- Choose the size from the benchmark results. Size 8 below is an example; use 4 if it performs better. The launcher requires a matching completed benchmark and an estimated generation time no greater than 90 hours. It refuses launching if the code, inputs, or selected size no longer match the benchmark.
- Production runs both shards in the foreground, verifies all outputs, then exits. Do not use the old detached `start-sequential` command inside a training job.
- Run:ai's default training job is preemptible. A full GPU request does not guarantee uninterrupted access or a completion date. The launcher leaves the cluster's scheduling priority unchanged.
- A 94-hour runtime limit plus a two-minute forced-termination grace keeps the submitted job below four days of running time. Automatic deletion occurs ten minutes after completion; it is separate from the runtime limit. Neither step deletes the existing PVC or previously generated images.

```bash
# Run only after inspecting the matching benchmark.
bash scripts/run_hismmdm_corners.sh submit-run 8
```

- If preempted or stopped before completion, resubmit with the same code, inputs, environment and batch size. New job records are separate, while valid production completion receipts are reused. Repeated attempts can exceed four days cumulatively; check the resource agreement before extending the total allocation period.
- Outputs are `fold-0-corners-16000/generated/shard-0` and `shard-1`. No automatic OneDrive upload occurs.

```bash
bash scripts/run_hismmdm_corners.sh status
bash scripts/run_hismmdm_corners.sh verify
```

## 5. QC, review, and handoff

- The existing pixel-only synthetic QC can read the generated manifests: shard identity, fold membership, native output path, 224-pixel classifier path and their checksums are preserved. Use the agreed explicit `--blur-reject-below 10` when that step is authorized; no thresholds are changed by this generator.
- This is pixel-only synthetic screening, not a new neural GrandQC run. Original WSI GrandQC masks are not evidence of synthetic image quality.
- Native inputs and outputs belong in side-by-side pathologist review at 128 × 128, with the 224-pixel original and marked crop position as context. The previous pathologist's general approval of resize-arm images does not constitute review of these new crop-arm images.
- Preserve crop/patient/fold lineage when splitting or sampling downstream. Four crops and their variants are correlated observations, not additional independent patients. Validation and external testing use real held-out images.
- Export/transfer only after recording QC and review outcomes. No acceptance rate, biological validity, or classifier improvement is assumed in advance.

## 6. Git and testing

- Commit this document, the crop Python and shell scripts, `tests/test_hismmdm_corners.py`, and `runai-full-gpu-check.json`; also commit the existing shared `his_mmdm_he_augment.py` dependency and the separately created `scripts/run_hismmdm_gpu_check.sh` after reviewing them. The JSON contains the user's reported infrastructure check and its exact scope. Keep large images, benchmark/run logs and authentication material outside Git.
- This change adds files only. It does not replace the existing core, batch runner, real-data QC policies, prior QC receipts, notebook, or old generation output.
- CPU tests cover exact crop pixels/overlap, duplicate and held-out lineage rejection, replay membership and checksum failures, deterministic batch seeds, and empty/missing completion receipts. CUDA sampling and real throughput must be validated on the user's full-GPU benchmark; they cannot be tested in the author's CPU-only environment.

```bash
python3 -m unittest discover -s tests -p test_hismmdm_corners.py -v
bash -n scripts/run_hismmdm_corners.sh
```
