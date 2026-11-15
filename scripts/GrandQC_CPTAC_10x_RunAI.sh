#!/usr/bin/env bash
# GrandQC from Weng et al., Nature Communications 2024. Academic/noncommercial.
# https://github.com/cpath-ukk/grandqc
set -o pipefail

BASE=${BASE:-/workspace/persist/crivas4-microscopy}
DATA="$BASE/GBM_MGMT"
ROOT="$DATA/grandqc"
TEAM="$BASE/gbm-mgmt-wsi"
CODE="$BASE/grandqc/01_WSI_inference_OPENSLIDE_QC"
GQC_PY="$BASE/grandqc-env/bin/python"
QC_WEIGHT="$CODE/models/qc/GrandQC_MPP1.pth"
TD_WEIGHT="$CODE/models/td/Tissue_Detection_MPP10.pth"
QC_MD5=ece5ac9967ec4d4320465ce0241de97f
TD_MD5=25ccb2592f5d44b4f30143501390e078
PILOT="$ROOT/tcga-halfA-mpp10-pilot"
CPTAC="$ROOT/cptac-mpp10"
DEST="$DATA/external/grandqc/CPTAC"
OLD="$ROOT/cptac-mpp15.FeYFeX"

verify_masks() {
  "$GQC_PY" - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from openslide import OpenSlide
from PIL import Image

inputs, masks = map(Path, sys.argv[1:])
Image.MAX_IMAGE_PIXELS = None
errors = []
files = sorted(inputs.glob("*.svs"))
unexpected = {p.name for p in masks.glob("*.svs_mask.png")} - {f"{p.name}_mask.png" for p in files}
if unexpected:
    raise SystemExit(f"Unexpected masks in {masks}: {', '.join(sorted(unexpected))}")
for path in files:
    try:
        with OpenSlide(str(path)) as slide:
            width, height = slide.dimensions
            expected = (width * float(slide.properties["openslide.mpp-x"]),
                        height * float(slide.properties["openslide.mpp-y"]))
        with Image.open(masks / f"{path.name}_mask.png") as mask:
            if mask.mode not in ("L", "P"):
                raise ValueError(f"unexpected image mode {mask.mode}")
            if any(abs(got / want - 1) > 0.01 for got, want in zip(mask.size, expected)):
                raise ValueError(f"size {mask.size}, expected {expected[0]:.0f}x{expected[1]:.0f}")
            if mask.getextrema()[1] > 7:
                raise ValueError("unexpected pixel class")
    except Exception as error:
        errors.append(path.name)
        print(f"{path.name}: {error}", file=sys.stderr)
print(f"Verified {len(files) - len(errors)}/{len(files)} masks at 1.0 µm/px.")
if not files or errors:
    raise SystemExit(1)
PY
}

prepare_weights() {
  if [ ! -s "$QC_WEIGHT" ] || ! printf '%s  %s\n' "$QC_MD5" "$QC_WEIGHT" | md5sum -c --quiet -; then
    mkdir -p "$(dirname "$QC_WEIGHT")" &&
    curl -fL --retry 3 \
      'https://zenodo.org/records/14041538/files/GrandQC_MPP1.pth?download=1' \
      -o "$QC_WEIGHT.part" &&
    printf '%s  %s\n' "$QC_MD5" "$QC_WEIGHT.part" | md5sum -c - &&
    mv -- "$QC_WEIGHT.part" "$QC_WEIGHT" || return 1
  fi
  printf '%s  %s\n' "$TD_MD5" "$TD_WEIGHT" "$QC_MD5" "$QC_WEIGHT" | md5sum -c -
}

stage_slides() {
  "$GQC_PY" - "$TEAM/manifests/collection_manifest.json" "$DATA" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

manifest, data, inputs = map(Path, sys.argv[1:4])
cohort = sys.argv[4]
pilot = {
    "TCGA-06-1084-01Z-00-DX2.f36f31e3-2d41-410c-a5b4-91f1f7508eec.svs",
    "TCGA-12-0820-01Z-00-DX1.43DEB9E2-D419-4263-8188-20FEBB070AAF.svs",
    "TCGA-41-4097-01Z-00-DX1.9d34ecf6-35db-4b8f-b7f8-f5ae4183a49c.svs",
}
rows = [r for r in json.loads(manifest.read_text())["files"] if r["source"] == cohort]
if cohort == "TCGA":
    rows = [r for r in rows if r["file_name"] in pilot]
expected = 3 if cohort == "TCGA" else 204
names = {row["file_name"] for row in rows}
if len(rows) != expected or len(names) != expected:
    raise SystemExit(f"Expected {expected} unique {cohort} slides.")
inputs.mkdir(parents=True, exist_ok=True)
if any(link.name not in names for link in inputs.iterdir()):
    raise SystemExit(f"Unexpected files in {inputs}.")
sources = {inputs / row["file_name"]: data / "raw" / cohort / row["patient_id"] / row["file_name"]
           for row in rows}
sizes = {row["file_name"]: row["size_bytes"] for row in rows}
missing = [str(src) for src in sources.values()
           if not src.is_file() or src.stat().st_size != sizes[src.name]]
if missing:
    raise SystemExit(f"Missing or wrong-size slides ({len(missing)}), first: {missing[0]}")
for dest, src in sources.items():
    if dest.is_symlink() and dest.resolve() == src.resolve():
        continue
    dest.symlink_to(src.resolve())
print(f"Staged {expected} {cohort} slide links.")
PY
}

run_cohort() (
  cohort="$1"
  work="$2"
  inputs="$work/input"
  out="$work/output"
  cd "$CODE" || exit 1
  if ! grep -Fq 'tis_det_map.convert("L").resize(' main.py; then
    printf '%s\n' 'Restore the Pillow tissue-mask mode fix used in the previous run.' >&2
    exit 1
  fi
  prepare_weights && stage_slides "$inputs" "$cohort" && mkdir -p "$out" || exit 1
  pending="$(mktemp -d "$work/pending.XXXXXX")" || exit 1
  trap 'rm -rf -- "$pending"' EXIT
  mkdir "$pending/tissue" "$pending/artifact" || exit 1
  for slide in "$inputs"/*.svs; do
    name="${slide##*/}"
    if [ ! -s "$out/tis_det_mask/${name}_MASK.png" ]; then
      ln -s -- "$slide" "$pending/tissue/$name" || exit 1
    fi
  done
  if compgen -G "$pending/tissue/*.svs" >/dev/null; then
    "$GQC_PY" -u wsi_tis_detect.py --slide_folder "$pending/tissue" --output_dir "$out" \
      2>&1 | tee -a "$out/tissue.log" || exit 1
  fi
  for slide in "$inputs"/*.svs; do
    name="${slide##*/}"
    if [ ! -s "$out/tis_det_mask/${name}_MASK.png" ]; then
      printf 'Missing tissue mask for %s\n' "$name" >&2
      exit 1
    fi
    if [ ! -s "$out/mask_qc/${name}_mask.png" ]; then
      ln -s -- "$slide" "$pending/artifact/$name" || exit 1
    fi
  done
  if compgen -G "$pending/artifact/*.svs" >/dev/null; then
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$GQC_PY" -u main.py \
      --slide_folder "$pending/artifact" --output_dir "$out" \
      --mpp_model 1.0 --create_geojson N 2>&1 | tee -a "$out/artifacts.log" || exit 1
  fi
  verify_masks "$inputs" "$out/mask_qc"
)

case "${1:-}" in
  pilot)
    run_cohort TCGA "$PILOT" &&
    tar -cf "$PILOT/tcga-halfA-mpp10-mask_qc.tar.part" -C "$PILOT/output" mask_qc &&
    tar -tf "$PILOT/tcga-halfA-mpp10-mask_qc.tar.part" &&
    mv -- "$PILOT/tcga-halfA-mpp10-mask_qc.tar.part" "$PILOT/tcga-halfA-mpp10-mask_qc.tar" &&
    ls -lh "$PILOT/tcga-halfA-mpp10-mask_qc.tar"
    ;;
  cptac)
    run_cohort CPTAC "$CPTAC" &&
    mkdir -p "$DEST" &&
    cp -p -- "$CPTAC/output/mask_qc/"*.svs_mask.png "$DEST/" &&
    printf 'Published the verified masks to %s\n' "$DEST"
    ;;
  delete-old)
    rm -rf -- "$OLD" &&
    rm -f -- "$OLD.tar" "$OLD.tar.part" &&
    printf 'Removed the previous 7× run and its archive files.\n'
    ;;
  *)
    printf 'Usage: bash %s {pilot|cptac|delete-old}\n' "$0" >&2
    exit 2
    ;;
esac
