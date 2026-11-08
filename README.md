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
| `--register` | Verify slides already in `raw/` and write their receipts; download nothing (see below). |
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

### Registering slides you already have

Slides copied from elsewhere (another machine, a teammate's Drive) have no receipts, so `tile.py`
treats them as not downloaded. Put them at `raw/{TCGA,CPTAC}/<patient_id>/<file>.svs` and run:

```
python data_prep/download.py --root /path/to/data --subset A --register
```

Each file gets the same checks as a download (size, TIFF/SVS header, and the GDC MD5 for TCGA)
and, if it passes, the same receipt. CPTAC has no publisher checksum, so its SHA-256 is computed
and recorded with `publisher_md5_checked: false`; that hash detects later changes but does not prove
the copy matches the source. A file that fails is left in place and reported. Nothing is downloaded.

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

`data_prep/tile.py` cuts each downloaded slide into tissue tiles. The tile grid and tile reads are
TIAToolbox's `SlidingWindowPatchExtractor`; the script adds the tissue mask, per-tile scores, lossless
storage, receipts, an integrity audit and parallel processing. It reads, scores and records **every**
tissue tile, stores the pixels of all of them or of a seeded sample per slide (`max_tiles_per_slide`), and
makes no quality decisions: those come from `configs/qc_policy.yaml` (see below), so changing a QC
cutoff never means re-tiling. The real-patch cache feeds both training conditions (real-only and
real + synthetic); synthetic patches are generated separately and matched to the same 224 px,
0.5 µm/px geometry.

The settings live in `configs/tiling.yaml`:

| Setting | Value | Meaning |
|---|---|---|
| `tile_px`, `target_mpp` | 224, 0.5 | Every tile is 224 × 224 px at exactly 0.5 µm/px (112 µm). The grid has no overlap (stride = 224 px) and uses only complete windows: no padded tiles at the right and bottom margins. |
| `max_upsample_pct` | 1.0 | A slide coarser than 0.5 µm/px is enlarged when read; more than 1% fails the slide. The coarsest TCGA slides (0.504 µm/px) need 0.8%. This is an engineering tolerance, not a validated image-quality cutoff. |
| `mask_mpp` | 8.0 | The tissue mask is computed on a thumbnail at 8 µm/px. |
| `mask_method` | saturation_multiotsu | Tissue is colored and background is white or gray, so the mask thresholds the HSV saturation channel (median-blurred, `mask_median_px`) using the lower cut of a three-class Otsu, so pale tissue counts as tissue, then closes small gaps (`mask_close_px`, 0 = off). `saturation_otsu` is the two-class version for comparison; `otsu_gray` uses TIAToolbox's grayscale `OtsuTissueMasker`. |
| `mask_min_saturation` | 8 | The mask threshold is never below 8 (0 = no floor). On very faintly stained slides the three-class cut falls inside the background noise (as low as 2) and the mask would cover the glass. Each slide records whether the floor was applied. |
| `min_tissue_fraction` | 0.5 | A grid tile is tissue if more than half of it is masked (or all of it), the same rule as TIAToolbox's `min_mask_ratio`. |
| `max_tiles_per_slide`, `seed` | null, 0 | Store pixels for at most this many tissue tiles per slide (`null` = all). Above the cap, a random sample is drawn with a per-slide seed (hash of `seed` and the slide ID), so reruns and other machines store the same tiles. Every tissue tile is still read, scored and recorded, and any tile not stored can be reread exactly from the slide at its saved coordinates. Set this before a full run: storing every tile of subset B is projected at about 330 GB. |
| `glass_*`, `near_white_value`, `near_black_value` | 8 / 200, 250, 5 | Score definitions: a glass pixel has saturation < 8 and brightness > 200; near-white / near-black pixels have every channel ≥ 250 / ≤ 5. |
| `png_compress_level` | 1 | Tiles are stored as PNG, which is lossless at every level; the level only trades speed for file size. |

Tiles are resampled with TIAToolbox's default interpolation: area averaging when shrinking (the 40×
slides), cubic when enlarging. Every setting is checked for type and range before any slide is
processed, and a bad value stops the run with exit code 2.

Outputs go to `patches/<version>/` (here `patches/tiling-v7/`), so a new config version never
overwrites an old one. Change any setting, or `CODE_VERSION` in `tile.py`, and you must bump `version`:
`tile.py` refuses to add slides to a folder that already holds slides from another config or code.

```
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --dry-run
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --masks-only --slides C3N-02786-23
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --slides C3N-02786-23
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B
python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --audit
```

| Option | Meaning |
|---|---|
| `--root`, `--subset` | Same as download.py (required). Only downloaded and verified (or registered) slides are tiled. |
| `--slides ID ...` | Only these slides, by slide ID or file-name prefix (for a pilot). |
| `--workers N` | Slides processed in parallel. Default: CPUs allocated to this container (its cgroup CPU quota, not the host's CPU count) minus one. Each worker uses one numerical thread. Benchmark a few slides before raising it. |
| `--masks-only` | Compare masks without tiling: see below. |
| `--audit` | Recheck tiled slides: see below. |
| `--sources`, `--config`, `--manifest`, `--log-dir`, `--dry-run` | As in download.py. |

Ctrl-C stops the run; finished slides are kept and the next run continues with the rest. On a rerun,
a slide is skipped only if its receipt matches the current config and `CODE_VERSION`, its `.h5` and
`_patches.csv` files have the sizes recorded in the receipt, and its QC image exists. Otherwise it is
tiled again.

### Comparing tissue masks (`--masks-only`)

For the mask pilot, `--masks-only` saves a 2 × 2 panel per slide in `patches/<version>/mask_review/`:
the thumbnail, then the mask outline from a two-class saturation Otsu, a three-class Otsu without
the floor, and a three-class Otsu with the config's floor. All three use the same thumbnail, median
blur and closing, so the panels differ only in the number of Otsu classes and the floor. Each panel
shows its cut, masked fraction and number of tissue tiles, and `mask_review_<subset>.csv` lists the
same numbers. Nothing is tiled. Review pale regions, tissue edges, small islands and background on
these panels before choosing the production rule; a larger mask alone is not better.

### Checking the outputs (`--audit`)

The rerun check only compares file sizes. `--audit` recomputes each slide's `.h5` and `_patches.csv`
SHA-256 against its receipt, decodes every stored PNG and compares the hash of its pixels with the
hash recorded when the tile was read from the slide. The result is saved as
`state/tiling/<version>/audit_<subset>_<timestamp>.json`, with the config hash and code version.
Exit code 1 means a mismatch or a selected slide that is not tiled. Run it after the full run and
before training.

### Tiling outputs

```
<root>/
  patches/<version>/{TCGA,CPTAC}/<patient_id>/<slide_id>.h5            stored tiles for one slide
  patches/<version>/{TCGA,CPTAC}/<patient_id>/<slide_id>_patches.csv   one row per grid tile touching the mask
  patches/<version>/{TCGA,CPTAC}/<patient_id>/<slide_id>_qc.png        thumbnail: tissue outline (green), sample of tiles (blue)
  patches/<version>/{TCGA,CPTAC}/<patient_id>/<slide_id>_mask.png      the tissue mask
  patches/<version>/index_<subset>.csv                                  one row per tiled slide
  patches/<version>/mask_review/                                        --masks-only panels and CSV
  state/tiling/<version>/{TCGA,CPTAC}/<slide_id>.json                   receipt: config hash, counts, file hashes
  state/tiling/<version>/audit_<subset>_<timestamp>.json                --audit results
```

Every grid position touching the mask is read and scored. **Tissue** tiles are stored (all of them, or the
seeded sample when `max_tiles_per_slide` is set); **edge** tiles (touching the mask, but not more than half
tissue) and tissue tiles outside the sample are scored but not stored, and can be reread from their
coordinates. **Background** tiles are only recorded in the `.h5` grid arrays.

Each `.h5` file holds, for the stored tissue tiles in grid order:

- `tiles`: PNG bytes, one per tile (224 × 224 RGB, lossless)
- `pixel_sha256`: SHA-256 of each tile's decoded RGB pixels, recorded when it was read
- `coords`: top-left x, y of each tile at 0.5 µm/px. These are TIAToolbox `coord_space="resolution"`
  coordinates: `wsi.read_rect((x, y), (224, 224), 0.5, "mpp", coord_space="resolution")` reads the
  same tile again.
- `coords_level0`: top-left of the full-resolution region TIAToolbox read for each tile
- `grid_index`: each tile's position in the full grid

and, for every grid position: `grid_coords`, `grid_role` (0 background, 1 edge, 2 tissue),
`grid_h5_index` (row in `tiles`, or -1 if not stored) and `grid_mask_fraction`. Its attributes record the slide's own µm/px and the upsampling applied, the tile
geometry, tile counts, the mask threshold used (`mask_threshold`), the Otsu cut before the floor
(`mask_otsu_threshold`), whether the floor was applied (`mask_floor_applied`), the full config and the
library versions, including the native OpenSlide library.

`_patches.csv` has one row per tissue or edge tile: patch ID, role, `stored`, `h5_index` (row in the `.h5`,
blank if not stored), grid index, coordinates at 0.5 µm/px and full resolution, mask fraction, the scores
(mean RGB, grayscale median, Laplacian variance, near-white, near-black and glass fractions) and the
pixel hash. Patch IDs (`<source>_<16 hex of SHA-256(slide_id)>_<x>_<y>`), column names and score
definitions match the team's `gbm_patching.py`, so results can be compared at matched coordinates.

A slide with no tissue tiles is still written with zero tiles and logged as a warning, so it shows up
in the index with `n_tissue_tiles` 0 and is not retried on every run. The index and receipts give
`n_tissue_tiles` (all tissue tiles) and `n_stored_tiles` (tiles with pixels in the `.h5`).

To read a tile:

```python
import h5py, io
from PIL import Image
with h5py.File("slide.h5") as h5:
    tile = Image.open(io.BytesIO(h5["tiles"][0].tobytes()))   # 224 x 224 RGB
    x, y = h5["coords_level0"][0]                              # full-resolution position
```

Exit codes: `0` every selected slide tiled (or audited clean) · `1` slides failed, not yet downloaded,
or failed the audit · `2` setup or config problem · `130` interrupted with Ctrl-C.

## Tile QC decisions

`data_prep/qc.py` applies `configs/qc_policy.yaml` to the scores in each slide's `_patches.csv` and
gives every tissue tile a decision, stored or not:

| Setting | Value | Meaning |
|---|---|---|
| `max_glass_fraction` | 0.9 | **Reject** a tile whose glass fraction is above 0.9. |
| `blank_fraction` | 0.99 | **Reject** a tile that is at least 99% near-white or near-black. |
| `dark_mean_rgb` | 50 | **Review** a tile whose mean R, G and B are all below 50 (`null` = off). |
| `min_blur` | null | **Review** a tile whose Laplacian variance is below this. Off until the team sets it: low values can mean blur or smooth, low-texture tissue. |

These are pilot values for team review, not validated cutoffs. Review cases need a person; only
`accept` tiles should be used until they are resolved. Bump the policy `version` when you change a
value.

```
python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B
```

It writes to `patches/<tiling version>/qc/<policy version>/`: `{source}/{patient_id}/{slide_id}_decisions.csv`
(patch ID, `stored`, `h5_index`, grid index, coordinates, decision, reasons; one row per tissue tile),
`slides_<subset>.csv` (counts and reasons per slide) and `policy_<subset>.json` (the policy, its hash
and the tiling index it was applied to).

## Repository layout

```
data_prep/download.py   download with resume and verification
data_prep/split.py      patient-level A/B split
data_prep/tile.py       tissue tiles with TIAToolbox
data_prep/qc.py         apply the tile QC policy
configs/tiling.yaml     tiling settings
configs/qc_policy.yaml  tile QC cutoffs
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
