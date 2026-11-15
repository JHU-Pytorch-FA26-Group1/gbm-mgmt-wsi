# Frozen HMU-1st H&E pilot on Run:ai

This implements the v0.2 feasibility protocol. It uses a few reviewed TCGA training-fold references and a frozen HMU-1st checkpoint. Review deterministic reconstructions before submitting eligible references to stochastic decoding. The outputs are for feasibility review.

## Verified release settings

| Item | Setting |
| --- | --- |
| His-MMDM revision | `26a1f9f12a8a3d1799496fb7dacebd722d92f3e3` |
| Official checkpoint archive | 6,897,817,378 bytes. Publisher MD5 `7354dfba1481afea9af0672476a04ca8`. |
| HMU-1st member | `pretrained_checkpoints/IHC_virtual_staining/diffusion_model308000.pt` |
| Extracted weights | 1,682,227,495 bytes. Measured SHA-256 `a2dd28c95984770e30ef30e90b9ea95d5bbc3f3ed5313dc44b936335233defc3`. |
| Source and target domain | Release token `HE`, class ID 10 of 26. Passed separately to the inversion and decoding loops. |
| DDIM schedule | 1,000 linear diffusion steps, no timestep respacing in this launcher. Invert at `eta=0`, with clipping disabled per upstream. Decode with clipping enabled per upstream. |
| Gradient mode | The wrapper keeps the released class-conditioned U-Net and disables the auxiliary classifier guidance in both phases. The stock decoder computes classifier input gradients. This choice implements the protocol's gradient-free requirement and is logged. |
| Input geometry | Source study tile is raw 224×224 RGB at 0.5 µm/px. For these square tiles the released transform resizes the full field to 128×128 using bicubic and scales RGB to [−1, 1]. Native output remains 128×128. The 112 µm field of view is preserved with effective 0.875 µm/px sampling. |
| Provenance boundary | The paper reports HMU-1st data from Harbin. The author's statement about no TCGA-GBM/CPTAC-GBM overlap is reported from their correspondence. Patient-level identity across sources remains unaudited. |

The paper reports 256-pixel windows for 20× slides and 512-pixel windows for 40× slides, followed by resizing to 128. The reported objective magnifications do not establish exact slide-level MPP for the released checkpoint. Compare generated images against the wrapper's **128×128 native input** to separate the observable effects of resampling from the reconstruction. The original 224×224 images remain available for morphology review.

Sources [His-MMDM paper, sections 2.2 and 4.2](https://doi.org/10.1002/advs.202518066) · [Pinned notebook and source](https://github.com/lzx325/His-MMDM/tree/26a1f9f12a8a3d1799496fb7dacebd722d92f3e3) · [HMU-1st virtual staining configuration](https://github.com/lzx325/His-MMDM/blob/26a1f9f12a8a3d1799496fb7dacebd722d92f3e3/scripts/dataset_specific_configurations/IHC_virtual_staining.sh)

## Run on the RTX PRO 6000

Run these commands from the team repository checkout. Use a fresh inference environment and your existing patch environment to read the modeling export. The following commands target the Blackwell GPU and use the CUDA 12.8 wheel pair already used for the GrandQC smoke test. The wrapper calls the Python sampler directly, so there is no MPI or cluster-specific launcher setup.

```bash
BASE=/workspace/persist/crivas4-microscopy
DATA="$BASE/GBM_MGMT"
TEAM="$BASE/gbm-mgmt-wsi"
PILOT="$DATA/hismmdm-pilot"
CODE="$BASE/His-MMDM"
SCRIPT="$TEAM/experiments/hismmdm-pilot/his_mmdm_he_augment.py"
PATCH_PY="$BASE/patch-env/bin/python"
HIS_PY="$BASE/hismmdm-env/bin/python"
REV=26a1f9f12a8a3d1799496fb7dacebd722d92f3e3

mkdir -p "$PILOT/artifacts"
if [ ! -d "$CODE/.git" ]; then
  git clone https://github.com/lzx325/His-MMDM.git "$CODE"
fi
git -C "$CODE" switch --detach "$REV" && git -C "$CODE" rev-parse HEAD

if [ ! -x "$HIS_PY" ]; then
  HIS_TOOLS="$BASE/.hismmdm-tools"
  env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT PIP_CONFIG_FILE=/dev/null \
    python3 -m pip install --target "$HIS_TOOLS/uv" uv &&
  env PYTHONPATH="$HIS_TOOLS/uv" UV_PYTHON_INSTALL_DIR="$HIS_TOOLS/python" \
    python3 -m uv venv --seed --python 3.10 "$BASE/hismmdm-env"
fi
his_pip() {
  env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT PIP_CONFIG_FILE=/dev/null \
    "$HIS_PY" -m pip "$@"
}
his_pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128 &&
his_pip install numpy==1.26.4 Pillow==10.4.0 pandas==1.5.3 scikit-image==0.25.2 tqdm==4.67.1 &&
"$HIS_PY" -c 'import torch, torchvision; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
```

Download just the official **checkpoint archive**. The extraction command writes only the one required diffusion checkpoint and its verification record. Budget about 8.6 GB for the archive plus extracted checkpoint. Reuse an existing verified archive if available.

```bash
ARCHIVE="$PILOT/artifacts/pretrained_checkpoints.tar.gz"
WEIGHTS="$PILOT/artifacts/hmu-1st"
if [ ! -s "$ARCHIVE" ]; then
  curl --fail --show-error --location --retry 3 --continue-at - \
    --output "$ARCHIVE.part" \
    https://hismmdm.s3.amazonaws.com/inference/pretrained_checkpoints.tar.gz &&
  mv "$ARCHIVE.part" "$ARCHIVE"
fi
"$HIS_PY" "$SCRIPT" weights --archive "$ARCHIVE" --output "$WEIGHTS"
```

### 1. Select and stage training-fold references

Start with one or two references with clear, interpretable tumor. For a larger pilot select up to 12 QC-approved patches across patients and MGMT groups. Include different morphology where already available. Use the existing approved patch list, existing per-patch QC measurements and blinded morphology review. Key the review join by `patch_id`. Use the verified TCGA patient labels to stratify selection after applying the fold restriction.

Prepare a CSV with the following header and one row per selected patch. Copy the four scores from team tile/QC data. Use `qc_defect_flags=none` for a clean, reviewed pilot tile. The QC-approved modeling export supplies the pixel hash, slide coordinates, source WSI path and final eligibility. The wrapper validates the numbers and requires a morphology note. The selection itself remains a human review.

```csv
patch_id,morphology,qc_defect_flags,tissue_fraction,laplacian_variance,near_white_fraction,near_black_fraction
```

The verified label CSV uses `patient_id,mgmt_status` with `methylated` or `unmethylated`. Only labels of selected TCGA training patients are needed.

Set `MODELING` to the actual fold-specific QC export, package or combined modeling folder and supply the two CSVs. Pass the fold **held out for validation**. The existing team export interface at `5d2f7dc` records that fold. For a partial QC export, pass `--allow-partial` to `prepare`. Every selected ID still has to pass the fold, export and image checks.

```bash
FOLD=0
MODELING="$DATA/packages/fold-0-combined"
LABELS="$DATA/manifests/tcga_patient_mgmt.csv"
SELECTION="$PILOT/smoke_reference_selection.csv"
REFS="$PILOT/fold-0-smoke-refs"

"$PATCH_PY" "$SCRIPT" prepare --team "$TEAM" --modeling "$MODELING" \
  --select "$SELECTION" --labels "$LABELS" --fold "$FOLD" --output "$REFS"
```

### 2. Dry run and generate deterministic controls

Inspect `preflight.json` for both HE IDs, the input geometry, the exact schedule and the reference list boundary. Use distinct output directories. Each actual run performs its own preflight again.

```bash
RUN=( "$HIS_PY" -u "$SCRIPT" run --upstream "$CODE" --weights "$WEIGHTS"
      --refs "$REFS" --fold "$FOLD" --device cuda:0 )
CTRL="$PILOT/fold-0-smoke-controls"

"${RUN[@]}" --phase reconstruct --dry-run --output "$PILOT/fold-0-smoke-preflight" &&
"${RUN[@]}" --phase reconstruct --output "$CTRL"
```

The reconstruction phase inverts once per reference and runs two `eta=0` decodes with different seeds. It checks equality before writing the control and saves the encoded reference with its SHA-256. Inspect `$CTRL/montages` and compare against `$REFS/originals` when useful. Use SSIM as a descriptive check alongside morphology.

### 3. Review, then run stochastic decoding

Copy `$CTRL/reconstruction_review.csv` to a review filename. Fill in `decision` with `pass` or `fail` and enter `reviewer` on every row. Keep the control hashes intact. If the reconstructions show systematic morphology changes, stop the pilot. The variants command checks the decision sheet and generates outputs only for passing references. Choose the same GPU type and environment used for the controls.

For an initial timing run, use `--etas 0.3`. For the small feasibility sweep use the exploratory values below. The script checks exact same-seed replay at the first seed for each eta and checks that the two seeds yield different RGB pixels. It reuses the exact saved inversion. Reviewers can inspect the montages without the generation manifest to stay blinded to eta, seed and MGMT.

```bash
REVIEW="$PILOT/fold-0-smoke-reviewed.csv"

"${RUN[@]}" --phase variants --reconstruction "$CTRL" --review "$REVIEW" \
  --etas 0.1 0.3 0.5 --seeds 101 102 --output "$PILOT/fold-0-smoke-variants"
```

When expanding to the 8–12 patch pilot, use a new selection CSV, stage directory, reconstruction directory and review sheet. Keep the scope within one working day. A positive result supports a separate decision about downstream experiments. Keep test and validation results out of pilot settings and continue the real-only plan if the pilot is inconclusive.

## Output checks and validation status

- `completed.json` records completion, run settings and environment. Record human approval in the reconstruction review sheet.
- `generation_manifest.csv` carries the patch lineage, sample seeds, hashes, QC selection information and SSIM scores. `run_log.jsonl` records per-reference completion or failure. Keep the manifest separate from blinded morphology review.
- Reconstructions and stochastic variants require qualified morphology review to support a fidelity finding. SSIM and seed checks measure software behavior and image similarities.
- Verified locally with the actual archive and diffusion weights. Archive MD5 and checkpoint SHA-256 matched. A strict CPU state-dict load succeeded, and a single-step CPU test with the real U-Net and upstream sampler checked the effect of positive eta. An HDF5 fixture exercised the wrapper's preflight, fold filtering, checksum refusal and reconstruction review gate. The 1,000-step GPU path and real GBM morphology review are pending.
