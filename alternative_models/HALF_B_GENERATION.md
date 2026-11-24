# Half B generation matched to Half A

## Comparison contract

- Use the verified `packages/B_qc-v5` package, TCGA tissue and fold-0 training patients only. Exclude validation patients, CPTAC and all Half A source identities. Halves A/B are processing partitions, not train/test partitions. Synthetics remain tiles of the original patient, never additional patients.
- Choose one pool of 2,000 original tiles, 1,000 per MGMT class, using Half A's original patient-quota, slide/block sampling and repeat-check selection functions. Both designs use this identical pool. Half B has 55 eligible training patients (21 methylated, 34 unmethylated), compared with 63 in Half A; therefore per-patient allocations differ and are recorded in `patient_allocation.csv`. Insufficient capacity is refused rather than reducing the budget or duplicating source IDs.

| Setting | Matched Half B rule |
|---|---|
| Original pixels | Unchanged package RGB 224 x 224 at 0.5 um/pixel |
| Resize | Original variants 01/02, then extension 03--08; exact original engines/seeds; 16,000 raw outputs |
| Corners | Four 128 x 128 crops at (0,0), (96,0), (0,96), (96,96); 32-pixel axis overlap; two variants/corner; 16,000 raw outputs |
| Sources | Same 2,000 patch IDs and original pixel hashes in both designs |
| Checkpoint | Same verified frozen HMU-1st weights and released upstream revision; H&E-to-H&E; no classifier guidance |
| Sampling | Same 1,000 DDIM steps, encode eta=0, decode eta=0.3, clipping/FP16 behavior and repeat/control checks |
| Resize batch | One source, as in Half A |
| Crop batch | Inherited from A's actual production record, not guessed or benchmark-selected again |
| Classifier copies | Same 128-to-224 bicubic interpolation |
| Synthetic QC | Original `synthetic-light-v4`, blur reject below 10, other thresholds unchanged, on existing 224-pixel copies |
| Review | Pending; QC acceptance does not certify biological or MGMT-label preservation |

- Each design yields 8,000 raw outputs per source-patient MGMT class before QC. Retention can differ across halves/designs. Preserve all raw outputs and actual accept/review/reject counts; no automatic threshold relaxation occurs.
- Resize retains the 112-um original field of view; corners retain 64 um. Upsampling does not restore resolution or enlarge the tissue field. These are intended design differences. MGMT is inherited patient metadata, not the generator condition.
- Generation remains **fold 0 only**. Do not reuse these variants as training inputs for another fold without rechecking the source patient's membership. Five-fold generation is not implemented here.

## Frozen engines and portable-package evidence

- Preparation locates inference files by the SHA-256 values in A's plans/run records and copies them unchanged into B's `templates/`. Crop receives its own recorded wrapper revision, which can differ from resize. Inference functions, seed mapping and reconstruction/replay checks are reused unchanged.
- An explicit `PortableCore` dependency adapter replaces only the legacy resize source-validation gate. Its hash is recorded in the protocol/source plan and immutable job snapshots. No inference function is replaced, no original source file is edited and existing model-owner implementations remain unchanged.
- Portable packages omit original whole-slide score CSVs. B records package approval, original modeling rows, package/fold/label hashes and every selected PNG/pixel hash. It neither invents mask/tissue fractions nor reruns a different masking method. This is a provenance adaptation, not a changed threshold or source filter.
- Preparation records Python/NumPy/pandas/Pillow versions and loader/common hashes. GPU workers require A's software/GPU-type runtime for their design. QC requires A's original policy, tiling, code hashes and QC runtime. Drift stops rather than silently changing the experiment.

## Install, test and commit

- Apply the new incremental patch once in the existing alternative worktree. The tests use real HDF5/PNG fixtures and a complete 2,000-source/8,000-crop preparation fixture; they do not run GPU diffusion. The four existing A engines must be available at the indicated path.

```bash
BASE=/workspace/persist/crivas4-microscopy
ALT="$BASE/gbm-mgmt-wsi-alternative-models"
cd "$ALT"
git apply --check "$BASE/hismmdm-half-b-matched32000.patch"
git apply "$BASE/hismmdm-half-b-matched32000.patch"
env HIS_B_TEST_TEMPLATES="$BASE/gbm-mgmt-wsi/experiments/hismmdm-pilot" \
  LD_PRELOAD="" LD_LIBRARY_PATH="$BASE/patch-env/lib/python3.12/site-packages/torch/lib" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$BASE/patch-env/bin/python" -m unittest discover \
  -s alternative_models/tests -p test_half_b_generation.py -v
bash -n scripts/run_hismmdm_half_b.sh
```

- Use the existing GitHub CLI credential helper, never a GitHub password. Commit only the new wrapper, tests, launcher and documentation; no merge is performed.

```bash
export GH_CONFIG_DIR="$BASE/github-cli-config"
GH="$(command -v gh || find "$BASE" -maxdepth 5 -type f -name gh -executable -print -quit)"
test -n "$GH"
"$GH" auth setup-git --hostname github.com
FILES=(alternative_models/half_b_generation.py alternative_models/tests/test_half_b_generation.py \
  alternative_models/HALF_B_GENERATION.md alternative_models/README.md scripts/run_hismmdm_half_b.sh)
git add -- "${FILES[@]}"
git diff --cached --check -- "${FILES[@]}"
git commit --only -m "Add matched Half B generation using recorded Half A engines" -- "${FILES[@]}"
GIT_TERMINAL_PROMPT=0 git push
```

## Prepare and record the shared source pool

- Preparation is CPU-only and submits no workload; it can run while A continues. Existing plans are refused. If only crop-file preparation was interrupted after source publication, use `finish-crops`; do not reselect/remove the source pool.

```bash
bash scripts/run_hismmdm_half_b.sh prepare
bash scripts/run_hismmdm_half_b.sh validate
ROOT="$BASE/GBM_MGMT/hismmdm-pilot/half-B-fold-0-matched-32000"
cat "$ROOT/resize-original-4000/patient_allocation.csv"
```

- `record` copies provenance and selection metadata into Git, never images/weights. It refuses differing existing records. Later invocations add final receipts without duplicating unchanged source records.

```bash
bash scripts/run_hismmdm_half_b.sh record
git add -- alternative_models/records/half-B-fold0
git diff --cached --check -- alternative_models/records/half-B-fold0
git commit --only -m "Record the matched Half B source pool and protocol" -- alternative_models/records/half-B-fold0
GIT_TERMINAL_PROMPT=0 git push
```

## Hand off GPU allocations after A finishes

- Check A's persistent completion/progress and the live scheduler. If authentication expired, run `runai login sso` and repeat the scheduler command. Running jobs do not require a live CLI login.

```bash
bash scripts/run_hismmdm_half_b.sh a-status
export PATH="$BASE/bin:/root/.runai/bin:$PATH"
export RUNAI_CLI_CONFIG_PATH="$BASE/runai-config"
runai training standard list --project runai-users
```

- Submit designs separately. Each command requires its corresponding A design to be finalized and its workload allocation released. Active/queued duplicates, unresolved A jobs and exceeding the 1.5-GPU generation budget are refused. No other user's job is changed or stopped.
- Crop requests one full GPU; resize requests half a GPU. Both use A's image/PVC/CPU/memory requests, a finite 94-hour timeout and automatic deletion ten minutes after completion. The problematic fractional GPU-limit flag is omitted.

```bash
bash scripts/run_hismmdm_half_b.sh submit-crop
bash scripts/run_hismmdm_half_b.sh submit-resize
```

- These commands explicitly submit jobs; no automatic handoff is installed. A blocked command submits nothing and can be rerun after its stated gate passes. Each job runs shard 0 then shard 1. Resize first generates 4,000 variants, then the matched 12,000 extension. Valid completion receipts are reused.
- **94 hours is a per-job upper limit, not a finish-time promise.** The complete resize design has more work than A's 12,000-output extension. Timeout/preemption may require another finite submission after the old allocation is gone. Keep all files and rerun the same submit command, never prepare a replacement pool.
- Each job verifies generation and then performs CPU blur-10 QC. If only QC was interrupted, run `qc-crop` or `qc-resize`; a new GPU job is unnecessary. Incomplete QC output is preserved under an `.incomplete-*` name; original images are untouched.

## Monitor, review and record final results

- Status reads persisted receipts, not live scheduler state. It reports all phases/shards against the complete 16,000-per-design budget. Per-shard progress JSON reports timing/update age; those ETAs exclude queueing, preemption, later QC and transfer.

```bash
bash scripts/run_hismmdm_half_b.sh status
runai training standard list --project runai-users
find "$ROOT/runai" -name runtime.log -print
```

- `crop-generation-complete.json` / `resize-generation-complete.json` mark finalized generation. `crop-qc-complete.json` / `resize-qc-complete.json` summarize screening of all 16,000 images per design. Detailed accepted/review tables and per-shard receipts are under `$ROOT/qc/synthetic-light-v4/`.
- Originals, native inputs, eta-zero controls, native variants and 224-pixel classifier copies are preserved in their phase folders. Compare original/input/native generated images for pathology review. Controls/copies are not extra variants. New designs/patients still require review before modeling; no training eligibility is automatically granted.
- Run `record` again after QC and commit newly copied final receipts. Existing source records must remain byte-identical.

```bash
bash scripts/run_hismmdm_half_b.sh record
git add -- alternative_models/records/half-B-fold0
git diff --cached --check -- alternative_models/records/half-B-fold0
git commit --only -m "Record completed Half B generation and blur-10 QC" -- alternative_models/records/half-B-fold0
GIT_TERMINAL_PROMPT=0 git push
```

## Archive and transfer

- Only after both QC summaries are complete, archive the entire B experiment: sources, frozen engines, records, all raw outputs and QC tables. Preserve rejection/review metadata. This does not replace/delete old archives or source tiles. The command block stops if either design lacks its completed QC summary.

```bash
(
set -euo pipefail
python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
for arm in ('crop', 'resize'):
    r = json.loads((Path(sys.argv[1]) / f'{arm}-qc-complete.json').read_text())
    if not r['complete'] or r['inputs'] != 16000 or sum(r['counts'].values()) != 16000:
        raise SystemExit(f'Incomplete QC: {arm}')
PY
FILE="HisMMDM_HalfB_matched32000_lightQC_v4_$(date -u +%Y%m%d-%H%M%S).tar"
ARCHIVE="$BASE/packages/$FILE"
tar --exclude='*/.worker.lock' --exclude='*/.crop.worker.lock' --exclude='*/.resize.worker.lock' \
  -cf "$ARCHIVE.part" -C "$BASE/GBM_MGMT/hismmdm-pilot" half-B-fold-0-matched-32000
mv "$ARCHIVE.part" "$ARCHIVE"
(cd "$BASE/packages" && sha256sum "$FILE" > "$FILE.sha256")
export PATH="$BASE/bin:$PATH"
export RCLONE_CONFIG="$BASE/rclone-config/rclone.conf"
export RCLONE_DISABLE_HTTP2=true
export RCLONE_MULTI_THREAD_STREAMS=0
DEST="jhu-onedrive:GBM_MGMT/hismmdm-pilot"
rclone copy "$BASE/packages" "$DEST" \
  --include "/$FILE" --include "/$FILE.sha256" --checksum --transfers 1 --progress \
  --log-file "$BASE/packages/${FILE%.tar}.upload.log" --log-level INFO
rclone check "$BASE/packages" "$DEST" \
  --include "/$FILE" --include "/$FILE.sha256" --one-way
)
```

- Keep the archive, checksum, verified B package and original synthetics. HTTP/2 remains disabled for the OneDrive copy, matching the setting that completed B's real-tile download successfully. Review and later model integration remain explicit scientific steps.
