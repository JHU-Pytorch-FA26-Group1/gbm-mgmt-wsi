# GrandQC for CPTAC-GBM on Run:ai

Generate GrandQC artifact masks at **1.0 µm/px (10×)** for all **204 CPTAC slides from 83 patients** in the frozen manifest. Half A contains 107 CPTAC slides and Half B contains 97. The TCGA slides in this workflow serve only as a reproduction pilot for the published reference masks.

| Stage | Checkpoint | Resolution | Expected MD5 |
| --- | --- | --- | --- |
| Tissue detection | [Tissue_Detection_MPP10.pth](https://zenodo.org/records/14507273) | 10 µm/px | `25ccb2592f5d44b4f30143501390e078` |
| Artifact segmentation | [GrandQC_MPP1.pth](https://zenodo.org/records/14041538) | 1.0 µm/px | `ece5ac9967ec4d4320465ce0241de97f` |

The OpenSlide runner uses `--mpp_model 1.0 --create_geojson N`. GrandQC defaults to the 7× model (1.5 µm/px), so use the explicit value above. The [Zenodo mask record](https://zenodo.org/records/14041578) also lists 1.5 µm/px, but eight checked masks from its `GBM.tar` match 1.0 µm/px when measured against the matching SVS dimensions and native MPP. Three CPU pilot masks generated at 1.0 µm/px matched the reference dimensions exactly and agreed with 99.9956%–99.9996% of decoded class IDs across all pixels, including background and padding. The pilot establishes near agreement on those slides. [GrandQC model variants](https://github.com/cpath-ukk/grandqc#how-to-use-different-versions-5x-7x-10x)

## 1. Set paths and check inputs

```bash
export BASE=/workspace/persist/crivas4-microscopy
DATA="$BASE/GBM_MGMT"
TEAM="$BASE/gbm-mgmt-wsi"
GRANDQC="$BASE/grandqc"
CODE="$GRANDQC/01_WSI_inference_OPENSLIDE_QC"
GQC_PY="$BASE/grandqc-env/bin/python"
RUNNER="$TEAM/scripts/GrandQC_CPTAC_10x_RunAI.sh"
RUN="$DATA/grandqc/cptac-mpp10"
OUT="$RUN/output"
```

Use the [team downloader](README.md) if slides are missing. For a workspace with Half A already present, fetch Half B with `--subset B --sources CPTAC`. Use `--subset full --sources CPTAC` to fetch both halves. The runner checks the expected filename and byte size for every slide against `manifests/collection_manifest.json` and stages symlinks to `raw/CPTAC/<patient_id>/<file_name>`.

### Fresh workspace setup

Reuse the existing environment, checkout and weights if they already pass the checks below. For a fresh GrandQC checkout, pin the upstream revision used for the pilot.

```bash
if [ ! -d "$GRANDQC/.git" ]; then
  git clone https://github.com/cpath-ukk/grandqc.git "$GRANDQC" &&
    git -C "$GRANDQC" checkout --detach 002688d74a4ac86dfbb816a96df8461d6080f88a
fi
git -C "$GRANDQC" rev-parse HEAD
```

Create `$BASE/grandqc-env` with Python 3.10 and pip. For example, use `python3.10 -m venv "$BASE/grandqc-env"` if that interpreter is installed, or `uv venv --seed --python 3.10 "$BASE/grandqc-env"` if `uv` is available. For a fresh environment on the original Run:ai Blackwell setup, use the CUDA 12.8 PyTorch wheels. The wrapper clears the base image's pip pins for these installs.

```bash
gqc_pip() {
  env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT PIP_CONFIG_FILE=/dev/null \
    "$GQC_PY" -m pip "$@"
}

grep -v '^torch==' "$GRANDQC/requirements.txt" > "$BASE/grandqc-requirements-blackwell.txt" &&
  gqc_pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128 &&
  gqc_pip install -r "$BASE/grandqc-requirements-blackwell.txt" openslide-python openslide-bin
```

Check GPU execution before launching inference. The original environment reported PyTorch `2.8.0+cu128`, CUDA `12.8`, and a result of `2.0`.

```bash
"$GQC_PY" -c 'import torch, torchvision, openslide, segmentation_models_pytorch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print((torch.ones(1, device="cuda") + 1).item())'
```

Install the tissue checkpoint once. The runner verifies both checkpoints and fetches the 10× artifact checkpoint when needed.

```bash
TD="$CODE/models/td/Tissue_Detection_MPP10.pth"
mkdir -p "$(dirname "$TD")" &&
  curl -fL --retry 3 \
    'https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth?download=1' \
    -o "$TD.part" &&
  printf '%s  %s\n' 25ccb2592f5d44b4f30143501390e078 "$TD.part" | md5sum -c - &&
  mv -- "$TD.part" "$TD"
```

Apply the Pillow compatibility fix once. It converts the binary tissue mask to 8-bit grayscale before the upstream resize to avoid the `image has wrong mode` error.

```bash
"$GQC_PY" - "$CODE/main.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
original = "tis_det_map.resize("
fixed = 'tis_det_map.convert("L").resize('
if fixed in source:
    print("Mask-mode fix already present.")
elif source.count(original) == 1:
    path.write_text(source.replace(original, fixed, 1))
    print("Applied mask-mode fix.")
else:
    raise SystemExit("Could not locate the expected tissue-mask resize.")
PY
```

## 2. Replay the TCGA pilot

```bash
bash "$RUNNER" pilot
```

This stages three Half A slides from `raw/TCGA`. The output archive is `GBM_MGMT/grandqc/tcga-halfA-mpp10-pilot/tcga-halfA-mpp10-mask_qc.tar`. Compare decoded class IDs at matching coordinates with the three reference PNGs in the [published TCGA archive](https://zenodo.org/records/14041578). Record the dimensions and mismatch counts separately if replaying on GPU.

| TCGA slide | Scan | CPU pilot mask size (W × H) | Different pixels from reference |
| --- | --- | --- | --- |
| TCGA-06-1084 DX2 | 20× | 8,031 × 7,054 | 252 / 56,650,674 |
| TCGA-12-0820 DX1 | 20× | 9,993 × 9,498 | 4,172 / 94,913,514 |
| TCGA-41-4097 DX1 | 40× | 5,319 × 16,957 | 3,451 / 90,194,283 |

The source patch width printed in the artifact log is approximately `512 × 1.0 / native_mpp` pixels. A source scanned at 0.494 µm/px should report about 1,036 pixels. At 0.25 µm/px, expect about 2,048 pixels.

## 3. Generate or resume CPTAC masks

```bash
bash "$RUNNER" cptac
```

Run one instance against this output directory at a time. The runner skips existing nonempty masks, runs tissue detection before artifact segmentation, and checks all 204 artifact masks against the staged slides for names, dimensions within 1%, and the expected class range. If verification flags a corrupt or wrong-resolution PNG, move that file out of the output directory and rerun the command.

| Location under `GBM_MGMT` | Contents |
| --- | --- |
| `grandqc/cptac-mpp10/output/tis_det_mask/` | Binary tissue mask PNGs |
| `grandqc/cptac-mpp10/output/mask_qc/` | Artifact class IDs in `<file_name>_mask.png` files, where `file_name` includes `.svs` |
| `grandqc/cptac-mpp10/output/overlays_qc/` | Overlay JPGs for visual review |
| `external/grandqc/CPTAC/` | Verified artifact masks for the team's patch QC importer |

From a second terminal, set the paths in step 1 and monitor the run. The estimate uses the most recent 15 artifact-mask writes and varies with slide size.

```bash
watch -n 15 -x python3 -c '
from datetime import timedelta
from pathlib import Path
import sys

root = Path(sys.argv[1])
tissue = sum(p.stat().st_size > 0 for p in (root / "tis_det_mask").glob("*.svs_MASK.png"))
times = sorted(p.stat().st_mtime for p in (root / "mask_qc").glob("*.svs_mask.png")
               if p.stat().st_size > 0)
done = len(times)
print(f"Tissue {tissue}/204 | Artifact {done}/204")
recent = times[-15:]
if 3 <= done < 204 and recent[-1] > recent[0]:
    seconds_per_slide = (recent[-1] - recent[0]) / (len(recent) - 1)
    print(f"Estimated artifact time remaining: {timedelta(seconds=round((204-done)*seconds_per_slide))}")
' "$OUT"
```

The logs are `$OUT/tissue.log` and `$OUT/artifacts.log`. Wait for `Verified 204/204 masks at 1.0 µm/px.` and `Published the verified masks to ...` in the launch terminal. The class PNGs use values 0–7, which look almost black in a grayscale viewer. Use the overlay JPG for visual inspection. [GrandQC class labels](https://github.com/cpath-ukk/grandqc#model-output-pixel-class-coding--class-labels)

## 4. Archive and hand off

After verification and publication, create a flat tar of the 204 artifact PNGs. The archive is checked against the source files before it receives its final filename.

```bash
(
  cd "$OUT/mask_qc" || exit 1
  shopt -s nullglob
  masks=(*.svs_mask.png)
  if (( ${#masks[@]} != 204 )); then
    printf 'Found %s of 204 masks.\n' "${#masks[@]}" >&2
    exit 1
  fi
  archive="$DATA/grandqc/CPTAC_GrandQC_10x_masks.tar"
  temp=$(mktemp "$archive.XXXXXX") || exit 1
  trap 'rm -f -- "$temp"' EXIT
  tar -cf "$temp" -- "${masks[@]}" &&
    tar -df "$temp" &&
    mv -- "$temp" "$archive" &&
    ls -lh "$archive"
)
```

Keep the run logs and provenance alongside the handoff. Record the team and GrandQC commits, the GrandQC source fix, Python dependencies, the runner and manifest hashes, both checkpoint hashes and the final tar checksum.

```bash
(
  cd "$DATA/grandqc" || exit 1
  git -C "$TEAM" rev-parse HEAD > "$RUN/team-commit.txt" &&
    git -C "$GRANDQC" rev-parse HEAD > "$RUN/grandqc-commit.txt" &&
    git -C "$GRANDQC" diff HEAD -- 01_WSI_inference_OPENSLIDE_QC/main.py \
      > "$RUN/grandqc-mask-mode.patch" &&
    "$GQC_PY" --version > "$RUN/python-version.txt" &&
    "$GQC_PY" -m pip freeze > "$RUN/grandqc-environment.txt" &&
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$RUN/gpu-driver.txt" &&
    sha256sum "$RUNNER" "$TEAM/manifests/collection_manifest.json" \
      "$CODE/models/td/Tissue_Detection_MPP10.pth" "$CODE/models/qc/GrandQC_MPP1.pth" \
      > "$RUN/run-inputs.sha256" &&
    sha256sum CPTAC_GrandQC_10x_masks.tar > CPTAC_GrandQC_10x_masks.tar.sha256
)
```

After transfer, run the first block from the directory holding the archive and its checksum. If the runner already published to this data root, proceed to the importer.

```bash
sha256sum -c CPTAC_GrandQC_10x_masks.tar.sha256 &&
  mkdir -p "$DATA/external/grandqc/CPTAC" &&
  tar -xf CPTAC_GrandQC_10x_masks.tar -C "$DATA/external/grandqc/CPTAC"
```

In the team's patching environment, import the masks for the tiled subset. Use `--subset A` or `--subset B` if only that half has been tiled.

```bash
python "$TEAM/data_prep/grandqc.py" --root "$DATA" --subset full --sources CPTAC --mask-mpp 1.0
```

The importer writes artifact fractions for patch QC. [Team importer documentation](README.md#grandqc-artifact-fractions-grandqcpy)
