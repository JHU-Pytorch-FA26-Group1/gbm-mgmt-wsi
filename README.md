# gbm-mgmt-wsi

Predicting MGMT promoter methylation in IDH-wildtype glioblastoma from H&E whole-slide images.
TCGA-GBM is the training and cross-validation cohort; CPTAC-GBM is held out for external evaluation.


## Data

No image data is stored in this repository. Slides are downloaded from the public sources below.

| Cohort | Source | Patients | Slides | Size |
|---|---|---|---|---|
| TCGA-GBM | [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-GBM) (diagnostic H&E SVS) | 151 | 320 | 149.9 GB |
| CPTAC-GBM | [TCIA PathDB](https://www.cancerimagingarchive.net/collection/cptac-gbm/) | 83 | 204 | 40.1 GB |

`manifests/collection_manifest.json` is the frozen slide list: one record per slide with patient ID,
slide ID, file name, download URL, byte size and, for TCGA, the GDC MD5. `download.py` refuses to run
unless this file's SHA-256 matches the hash pinned in the script, so everyone downloads the same slides.

## Environment

Local (conda):

```
conda env create -f environment.yml
conda activate gbm-mgmt
```

Google Colab (pip):

```
!pip install -r requirements.txt
```

Both install everything in `requirements.txt`, including TIAToolbox 2.1.3 for tiling. TIAToolbox is large
(it pulls in PyTorch) and pins versions of NumPy and PyTorch. On Colab, if you only need to download
slides, `!pip install requests tqdm` is enough and leaves Colab's own packages alone.

## Downloading slides

The download is split by patient into two subsets, A and B, so two people can each download about
95 GB. All slides from a patient stay in the same subset. The split is deterministic (`data_prep/split.py`,
version `gbm-team-halves-v1`) and does not use MGMT labels. It is a work split only, not a train/test split.

| Subset | Patients | Slides | Size |
|---|---|---|---|
| A | 117 | 265 | 95.0 GB |
| B | 117 | 259 | 95.0 GB |

Check the selection first, then download:

```
python data_prep/download.py --root ~/Bioinformatics/WSI/data --subset B --dry-run
python data_prep/download.py --root ~/Bioinformatics/WSI/data --subset B
```

On Colab, mount Drive in a notebook cell first and flush it afterwards:

```
from google.colab import drive; drive.mount('/content/drive')
!python data_prep/download.py --root /content/drive/MyDrive/GBM_MGMT --subset A --minutes 600
drive.flush_and_unmount()
```

| Option | Meaning |
|---|---|
| `--root` | Data folder (required). `raw/`, `state/` and `manifests/` are created inside it. |
| `--subset {full,A,B}` | Full cohort or one team subset (required). |
| `--sources TCGA CPTAC` | Cohorts to download, in this order (default: both). |
| `--minutes N` | Stop after N minutes (default: run until done). |
| `--max-mb N` | Stop after downloading N MB in this run, e.g. for a short pilot. |
| `--dry-run` | Show the selection, progress and disk space, then exit. |
| `--manifest`, `--log-dir` | Override the manifest and log locations. |

To stop, press Ctrl-C once. Rerunning the same command resumes from the last checkpoint and skips
verified slides.

### What the script checks

- **Every slide** must match its manifest size and have a TIFF/SVS header.
- **TCGA slides** must also match the GDC MD5. CPTAC does not publish checksums, so a SHA-256 is
  recorded at download time instead.
- **Partial downloads** are checkpointed every 64 MiB and on stop. A checkpoint is re-hashed before
  resuming, and a file that fails verification is kept as `.bad-*` and never marked complete.
- **The data folder** must not be inside a cloud-synced folder (Google Drive for desktop, iCloud
  Drive), and the disk must have room for the remaining download plus 10 GB.

### Outputs

```
<root>/
  raw/{TCGA,CPTAC}/<patient_id>/<file>.svs   downloaded slides
  state/{TCGA,CPTAC}/<slide_id>.verified.json   verification receipt per slide
  state/progress_half_<X>.csv   per-slide status (progress.csv for --subset full)
  state/transfer_log_half_<X>.jsonl   one line per download attempt
  manifests/   frozen manifest and the A/B patient and slide lists
```

Each run also writes a log file to `logs/download_<subset>_<timestamp>.log`.

Exit codes: `0` every selected slide verified · `1` slides still pending or failed (rerun to retry)
· `2` setup or source-metadata problem · `130` interrupted with Ctrl-C.

## Tiling slides into patches

`data_prep/tile.py` cuts each downloaded slide into tissue tiles. The tile grid, the tissue filter and
the tile reads are TIAToolbox's `SlidingWindowPatchExtractor`; the script adds the tissue mask, the capped
random sample, storage, receipts and parallel processing. The settings live in `configs/tiling.yaml`:

| Setting | Value | Meaning |
|---|---|---|
| `tile_px`, `target_mpp` | 256, 0.5 | Every tile is 256 × 256 px at exactly 0.5 µm/px (128 µm), matching PixCell-256. The grid has no overlap (stride = 256 px). |
| `mask_mpp` | 8.0 | The tissue mask is computed on a thumbnail at 8 µm/px. |
| `mask_method` | saturation_multiotsu | Tissue is colored and background is white or gray, so the mask thresholds the HSV saturation channel (median-blurred, `mask_median_px`), using the lower cut of a three-class Otsu so pale tissue counts as tissue, then closes small gaps (`mask_close_px`). `otsu_gray` uses TIAToolbox's grayscale `OtsuTissueMasker` instead. |
| `min_tissue_fraction` | 0.5 | A grid tile is kept if more than half of it is tissue. |
| `max_tiles_per_slide`, `seed` | 2000, 0 | Above the cap, tiles are sampled at random with a per-slide seed, so reruns pick the same tiles. |
| `max_glass_fraction`, `min_blur` | 0.9, 20 | Quality filter: a sampled tile is dropped if more than 90% of it is uncolored glass (saturation < 8, brightness > 200) or its blur score (variance of the Laplacian) is below 20, i.e. empty or featureless. Dropped tiles are replaced by the next tissue tile in the random order, so the cap is still reached. |
| `jpeg_quality` | 90 | Tiles are stored as JPEG. |

Tiles are resampled with TIAToolbox's default interpolation: area averaging when shrinking (the 40×
slides), cubic when enlarging. The file's SHA-256 is recorded in every output. Changing any setting
makes `tile.py` redo every slide; bump `version` when you do.

```
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --dry-run
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --slides C3N-02786-23
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B
```

| Option | Meaning |
|---|---|
| `--root`, `--subset` | Same as download.py (required). Only downloaded and verified slides are tiled. |
| `--slides ID ...` | Only these slides, by slide ID or file-name prefix (for a pilot). |
| `--workers N` | Slides processed in parallel (default: CPU count − 1). |
| `--sources`, `--config`, `--manifest`, `--log-dir`, `--dry-run` | As in download.py. |

Ctrl-C stops the run; finished slides are kept and the next run continues with the rest.

### Tiling outputs

```
<root>/
  patches/{TCGA,CPTAC}/<patient_id>/<slide_id>.h5       tiles for one slide
  patches/{TCGA,CPTAC}/<patient_id>/<slide_id>_qc.png   thumbnail: tissue outline (green), sampled tiles (blue)
  patches/index_<subset>.csv                            one row per tiled slide
  state/tiling/{TCGA,CPTAC}/<slide_id>.json             receipt: config hash, tile counts, file hash
```

Each `.h5` file holds:

- `tiles`: JPEG bytes, one per sampled tile
- `coords`: top-left x, y of each tile at 0.5 µm/px, in the same order. These are TIAToolbox
  `coord_space="resolution"` coordinates: `wsi.read_rect((x, y), (256, 256), 0.5, "mpp",
  coord_space="resolution")` reads the same tile again.
- `coords_level0`: top-left of the full-resolution region TIAToolbox read for each tile (for
  heatmaps and overlays)
- `tissue_coords`: `coords` for every tissue tile, including those not sampled
- `dropped_coords`: `coords` of tiles removed by the quality filter

Its attributes record the slide's own µm/px, the tile geometry, tile counts, the full config
and the library versions. To read a tile:

```python
import h5py, io
from PIL import Image
with h5py.File("slide.h5") as h5:
    tile = Image.open(io.BytesIO(h5["tiles"][0].tobytes()))   # 256 x 256 RGB
    x, y = h5["coords_level0"][0]                              # full-resolution position
```

Exit codes: `0` every selected slide tiled · `1` slides failed or not yet downloaded · `2` setup or
config problem · `130` interrupted with Ctrl-C.

## Repository layout

```
data_prep/download.py   download with resume and verification
data_prep/split.py      patient-level A/B split
data_prep/tile.py       tissue tiles with TIAToolbox
configs/tiling.yaml     tiling settings
manifests/              frozen slide manifest
logs/                   run logs (ignored by git; add submitted logs with git add -f)
environment.yml         conda environment
requirements.txt        pip requirements (also used by environment.yml)
```

## Status

- [x] Slide download and verification
- [ ] Tissue masking and patch extraction
- [ ] UNI2-h embeddings and PixCell synthetic bags
- [ ] Model training (ConvNeXt V2-Tiny, Swin V2-Tiny) and evaluation
