# Additional 4,000 resize variants from new training tiles

- Carlos selected a separate 0.5-GPU, auto-deleting training job and new training tiles. This adds 4,000 planned resize variants to the 4,000 already generated; it does not count any old variant twice or assume all new outputs will pass QC.
- Use 2,000 new source tiles, 1,000 from each patient MGMT status, with two variants each. Every original resize source ID is excluded before selection. The active crop arm uses those same old originals, so its source IDs are also excluded.
- New tiles can come from the same training patients. They are not new independent patients. Patient-balanced sampling uses as many eligible patients with remaining tiles as possible, without replacement.
- Keep held-out fold 0, TCGA training tissue only, Half A, the same verified package, labels, folds, tiling, modeling list and original source-score checks. Any changed package/configuration/labels stops preparation rather than silently changing the cohort. CPTAC and held-out TCGA patients cannot enter the source selection.
- Preserve full-field resize preprocessing: 224-pixel originals at 0.5 µm/px cover 112 µm, resized to 128 for the generator, with native 128-pixel outputs and separate 224-pixel bicubic classifier copies. This is not the corner-crop arm.
- Keep the existing frozen HMU-1st checkpoint and upstream revision, 1,000 DDIM steps, H&E-to-H&E class 10, inversion eta 0, generation eta 0.3, disabled classifier guidance, and the existing reconstruction/repeat checks. MGMT labels organize source quotas; they do not condition the generator.
- Use a recorded sampling seed equal to the original seed plus one by default (42 → 43 for the established run). Source IDs, pixels, patient/fold/class and output seeds remain in the manifests.
- New reference IDs continue after the original IDs (`ref002001`–`ref004000` for this extension), so output basenames do not collide with the original resize batch. Renaming leaves original PNG bytes/checksums unchanged and updates all affected manifest hashes before publication.

## 1. Code and environment

- This patch adds the preparer, launcher, this document and CPU tests. It does not modify the sampler or crop scripts. The existing `his_mmdm_batch.py` must be present from the working batch bundle and must be explicitly committed if it is still untracked. The shared core was committed with the crop work.
- Preparation wraps the legacy source selector only inside the preparation process to remove old source IDs, then calls the unchanged legacy preparation procedure. The selector is restored on success or error. No running process or source file is patched.
- Preparation uses `patch-env` Python 3.12 with its Torch library path. Inference uses the established `hismmdm-env` Python 3.10 with inherited library overrides cleared. No dependency upgrades are needed.
- Code was checked against the actual crop feature branch `c87f28a`; that branch's original/core files are retained. The legacy batch script is the existing working batch dependency; the launcher snapshots and hashes the actual file used on Run:ai.

## 2. Prepare new, disjoint source tiles

```bash
BASE=/workspace/persist/crivas4-microscopy
cd "$BASE/gbm-mgmt-wsi"
bash scripts/run_hismmdm_resize_extra.sh prepare
```

- Input parent: `GBM_MGMT/hismmdm-pilot/fold-0-balanced-4000`.
- New output: `GBM_MGMT/hismmdm-pilot/fold-0-resize-newtiles-4000`.
- Preparation stages its output in a temporary sibling and publishes it only after validating the selection and exclusions. It refuses to overwrite an existing output and does not redirect the original batch's `latest_batch.json`.
- `extension_receipt.json` records zero source overlap, parent/configuration hashes, code hashes, the new seed, class budgets, patient counts, and before/after selection counts. `references/excluded_parent_sources.csv` records the exact excluded IDs. If too few eligible new tiles remain, preparation fails instead of reusing old tiles or reducing the budget.
- This is a fresh source draw, not a matched-source crop-versus-resize comparison. For downstream experiments, distinguish the existing matched-source resize set from this additional-source set and control synthetic training counts when comparing strategies.

## 3. Submit the separate half-GPU job

```bash
bash scripts/run_hismmdm_resize_extra.sh submit
runai training standard list --project runai-users
```

- The job requests one device with GPU portion 0.5, four CPU cores and 16 GB CPU RAM. The optional --gpu-portion-limit is omitted because this cluster requires GPU Optimization for fractional limit settings. The prepared sources are retained; retry only submit. Confirm `0.50` allocated in the listing before interpreting it as running. It may queue; half a GPU does not guarantee twice the speed of the prior quarter-GPU job.
- It uses the confirmed image `nvcr.io/nvidia/pytorch:25.06-py3`, project `runai-users`, cluster `runai-cluster1` (selected explicitly before submission), and existing PVC `persistant-datasource-runai-cluster1-f06e9` at `/workspace/persist`.
- Submission records the CLI/Git versions and exact source snapshots and checksums under the new batch's `runai/<job>/` directory. It verifies the preparer, legacy batch and core still match the preparation receipt before submitting.
- The new job runs both shards sequentially in the foreground, verifies outputs, then exits. It is preemptible under the normal training policy. Completed source receipts enable resumption with matching code, inputs and runtime/GPU type. No extra parallel processes are started in the existing 0.25-GPU notebook workload.
- A 94-hour running-time limit plus a two-minute forced-termination grace bounds this attempt. Automatic deletion is ten minutes after completion. The existing PVC, the 4,000 old resize variants and the active 16,000-crop job are not deleted or reconfigured.
- Runtime output is saved directly to persistent `runai/<job>/runtime.log`; generation receipts/progress/manifests are in `generated/shard-0` and `generated/shard-1`. Capture the first measured rate before estimating completion. The historical quarter-GPU run took roughly two days for 4,000 variants; it is not a measured ETA for this new half-GPU job.

## 4. Monitor, resume, and review

```bash
watch -n 60 bash scripts/run_hismmdm_resize_extra.sh status
bash scripts/run_hismmdm_resize_extra.sh verify
```

- Resubmit with `submit` after a preemption/stop only once the previous job is no longer running. Existing shard locks protect workers from simultaneously writing the same shard. Keep the same code and inference setup; mismatched signatures stop rather than overwrite completed results.
- The job runs independently of the Jupyter browser tab. RL work in Colab uses a separate execution environment; no Colab configuration change is needed to keep the Run:ai job alive. Do not stop the Run:ai training job when switching to Colab.
- Generation completion does not imply QC acceptance or biological approval. Apply the agreed pixel-only synthetic QC with explicit blur cutoff 10 afterward, record decisions, and review the new images. No new neural GrandQC run, automatic OneDrive transfer, or automatic training approval is performed.
- Preserve both source sets, all raw generated outputs, QC sidecars, and review records. Before QC this extension would bring total resize outputs to 8,000, alongside 16,000 planned crop outputs.

## 5. Reproducibility checks

- CPU tests exercise exclusion before sampling, patient coverage and class budgets, exhaustion refusal, restoration of the selector after errors, and full preparation/lineage validation with a small verified package fixture. No GPU speed or new biological validity is claimed by these tests.

```bash
"$BASE/patch-env/bin/python" -m unittest discover -s tests -p test_hismmdm_resize_extension.py -v
bash -n scripts/run_hismmdm_resize_extra.sh
```

- Commit this document, the preparer, launcher and tests together with the unchanged legacy batch dependency. Keep data and logs on persistent storage, with source hashes and receipts for later packaging. Never commit CLI authentication configuration or bearer tokens.
