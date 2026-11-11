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

Both install everything in `requirements.txt`, including TIAToolbox 2.1.3 for tiling. The libraries that
decode, resample or normalize pixels (NumPy, OpenCV, Pillow, h5py, scikit-image, OpenSlide's Python
binding) are pinned to the versions the pilot ran with, so every machine that trains from a package uses the
same code; every output also records the versions it was made with. TIAToolbox is large (it pulls in
PyTorch) and pins versions of NumPy and PyTorch. On Colab, if you only need to download
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

## Processing pipeline

After download, each subset goes through these steps. `scripts/run_pipeline.sh` runs them in order with
one command and stops at the first step that fails:

| Step | Script | What it does |
|---|---|---|
| register | `download.py --register` | Optional: verify slides copied from elsewhere and write their receipts |
| masks | `tile.py --masks-only` | Optional: mask comparison panels for review; tiles nothing |
| tile | `tile.py` | Read, score and record every tissue tile; store the pixels of up to `max_tiles_per_slide` per slide |
| audit | `tile.py --audit` | Recheck every file, every stored tile's pixel hash and a sample of unstored tiles reread from the slide, and save the result |
| qc | `qc.py` | Apply `configs/qc_policy.yaml`: accept / review / reject per tissue tile |
| export | `export.py` | Apply human reviews and build `modeling_patches.csv` (needs a frozen policy) |

Then, outside the shell script: `package.py` copies an export and its model tiles into a portable folder,
and `combine.py` joins the packages of halves A and B (see "Packages for training").

```
scripts/run_pipeline.sh --root ~/Bioinformatics/WSI/data --subset B                 # tile, audit, qc, export
scripts/run_pipeline.sh --root ~/Bioinformatics/WSI/data --subset B --steps masks --workers 8
scripts/run_pipeline.sh --root /workspace/GBM_MGMT --subset A --steps register,tile,audit,qc
scripts/run_pipeline.sh --help
```

It uses the active Python environment (activate `gbm-mgmt` first), passes the failing step's exit code
through (`1` some slides or patients unfinished, `2` setup problem, `130` interrupted), and each step writes
its own log to `logs/`. `--dry-run` prints the commands without running them.

The code is split by job, with dependencies in one direction:

```
common.py   config checks, file writes and hashes, patch IDs, output paths, fold map
slides.py   slide opening, tissue mask, tile grid, tile roles, scores, cap sample (TIAToolbox)
tile.py     extraction command line: tiling, --masks-only, --audit
qc.py       QC policy -> decisions
review.py   human review records (patch, region, slide sign-off)
export.py   modeling list with per-patient accounting
package.py  portable copy of an export with exactly its model tiles
combine.py  join packages (halves A and B) into one model input
loader.py   for training code: read tiles, patient bags, Macenko per fold (not used by tiling)
```

## Tiling slides into patches

`data_prep/tile.py` cuts each downloaded slide into tissue tiles. The tile grid and tile reads are
TIAToolbox's `SlidingWindowPatchExtractor`; the mask, roles and scores are in `slides.py`. It reads, scores
and records **every** tissue tile, stores the pixels of all of them or of a seeded sample per slide
(`max_tiles_per_slide`), and makes no quality decisions: those come from `configs/qc_policy.yaml`, so
changing a QC cutoff never means re-tiling. The real-patch cache feeds both training conditions
(real-only and real + synthetic); synthetic patches are generated separately and matched to the same
224 px, 0.5 µm/px geometry.

The settings live in `configs/tiling.yaml`:

| Setting | Value | Meaning |
|---|---|---|
| `tile_px`, `target_mpp` | 224, 0.5 | Every tile is 224 × 224 px at exactly 0.5 µm/px (112 µm). The grid has no overlap (stride = 224 px) and uses only complete windows: no padded tiles at the right and bottom margins. |
| `max_upsample_pct` | 1.0 | A slide coarser than 0.5 µm/px is enlarged when read; more than 1% fails the slide. The coarsest TCGA slides (0.504 µm/px) need 0.8%. This is an engineering tolerance, not a validated image-quality cutoff. |
| `mask_mpp` | 8.0 | The tissue mask is computed on a thumbnail at 8 µm/px. |
| `mask_method` | saturation_multiotsu | Tissue is colored and background is white or gray, so the mask thresholds the HSV saturation channel (median-blurred, `mask_median_px`) using the lower cut of a three-class Otsu, so pale tissue counts as tissue, then closes small gaps (`mask_close_px`, 0 = off). `saturation_otsu` is the two-class version for comparison; `otsu_gray` uses TIAToolbox's grayscale `OtsuTissueMasker`. |
| `mask_min_saturation` | 8 | The mask threshold is never below 8 (0 = no floor). On very faintly stained slides the three-class cut falls inside the background noise (as low as 2) and the mask would cover the glass. Each slide records whether the floor was applied. |
| `min_tissue_fraction` | 0.5 | A grid tile is tissue if more than half of it is masked (or all of it), the same rule as TIAToolbox's `min_mask_ratio`. |
| `max_tiles_per_slide`, `seed` | 5508, 0 | Store pixels for at most this many tissue tiles per slide (`null` = all). 5,508 is the median number of tissue tiles per slide in subset B: about 100 GB for subset B, against about 230 GB for every tile. Above the cap, a random sample is drawn with a per-slide seed (hash of `seed` and the slide ID), so reruns and other machines store the same tiles. Every tissue tile is still read, scored, recorded and given a QC decision, and any tile not stored can be reread exactly from the slide. |
| `glass_*`, `near_white_value`, `near_black_value`, `dark_value`, `green_*` | | Score definitions: a glass pixel has saturation < 8 and brightness > 200; near-white / near-black pixels have every channel ≥ 250 / ≤ 5; a dark pixel has brightness < 60; a green pixel has hue 35–85 and saturation > 60 (OpenCV units). |
| `png_compress_level` | 6 | Tiles are stored as PNG, which is lossless at every level; the level only trades speed for file size. Level 1 makes files about 45% larger on Apple Silicon Macs. |

Tiles are resampled with TIAToolbox's default interpolation: area averaging when shrinking (the 40×
slides), cubic when enlarging. Every setting is checked for type and range before any slide is
processed, and a bad value stops the run with exit code 2.

Outputs go to `patches/<version>/` (here `patches/tiling-v7/`), so a new config version never
overwrites an old one. Change any setting, or `CODE_VERSION` in `common.py`, and you must bump `version`:
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
| `--slides ID ...` | Only these slides, by full slide ID or file-name prefix (for a pilot). A name that matches no slide in the subset stops the run (exit 2). For TCGA the slide ID is the full UUID; the file name starts with `TCGA-`. |
| `--workers N` | Slides processed in parallel. Default: CPUs allocated to this container (its cgroup CPU quota, not the host's CPU count) minus one. Each worker uses one numerical thread. Benchmark a few slides before raising it. |
| `--masks-only` | Compare masks without tiling: see below. |
| `--audit` | Recheck tiled slides: see below. |
| `--reread N` | With `--audit`: unstored tiles per slide to reread from the slide (default 5; 0 = none). |
| `--check-sources` | With `--audit`: also rehash every slide file against its download receipt (reads every slide in full). |
| `--sources`, `--config`, `--manifest`, `--log-dir`, `--dry-run` | As in download.py. |

Ctrl-C stops the run; finished slides are kept and the next run continues with the rest. On a rerun,
a slide is skipped only if its receipt matches the current config and `CODE_VERSION`, its `.h5` and
`_patches.csv` files have the sizes recorded in the receipt, and its QC image exists. Otherwise it is
tiled again.

### Comparing tissue masks (`--masks-only`)

For the mask pilot, `--masks-only` saves a 2 × 2 panel per slide in
`patches/<version>/mask_review/{source}/<file name>.png`: the thumbnail, then the mask outline from a
two-class saturation Otsu, a three-class Otsu without the floor, and a three-class Otsu with the config's
floor. All three use the same thumbnail, median blur and closing, so the panels differ only in the number of
Otsu classes and the floor. Each panel shows its cut, masked fraction and number of tissue tiles, and
`mask_review_<subset>.csv` lists the same numbers. Nothing is tiled. Review pale regions, tissue edges,
small islands and background on these panels before choosing the production rule; a larger mask alone is
not better.

### Checking the outputs (`--audit`)

The rerun check only compares file sizes. `--audit` recomputes each slide's `.h5` and `_patches.csv`
SHA-256 against its receipt, decodes every stored PNG and compares the hash of its pixels with the
hash recorded when the tile was read from the slide. Tiles that were not stored (edge tiles, tissue tiles
beyond the cap) reach the model only by rereading, so it also rereads a seeded sample of them per slide
(`--reread`, default 5) and checks their recorded hashes; this needs the slides. It also checks each slide's
mask and QC image against the hashes in its receipt, and with `--check-sources` rehashes the slide files
against their download receipts. The result is saved as
`state/tiling/<version>/audit_<subset>_<timestamp>.json`, with the config hash and code version.
Exit code 1 means a mismatch or a selected slide that is not tiled. `export.py` only uses slides whose
latest audit passed after they were tiled.

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
seeded sample of `max_tiles_per_slide`); **edge** tiles (touching the mask, but not more than half tissue)
and tissue tiles outside the sample are scored but not stored, and can be reread from their coordinates.
**Background** tiles are only recorded in the `.h5` grid arrays.

Each `.h5` file holds, for the stored tissue tiles in grid order:

- `tiles`: PNG bytes, one per tile (224 × 224 RGB, lossless)
- `pixel_sha256`: SHA-256 of each tile's decoded RGB pixels, recorded when it was read
- `coords`: top-left x, y of each tile at 0.5 µm/px. These are TIAToolbox `coord_space="resolution"`
  coordinates: `wsi.read_rect((x, y), (224, 224), 0.5, "mpp", coord_space="resolution")` reads the
  same tile again.
- `coords_level0`: top-left of the full-resolution region TIAToolbox read for each tile
- `grid_index`: each tile's position in the full grid

and, for every grid position: `grid_coords`, `grid_role` (0 background, 1 edge, 2 tissue),
`grid_h5_index` (row in `tiles`, or -1 if not stored) and `grid_mask_fraction`. Its attributes record the
slide's own µm/px and the upsampling applied, the tile geometry, tile counts, the mask threshold used
(`mask_threshold`), the Otsu cut before the floor (`mask_otsu_threshold`), whether the floor was applied
(`mask_floor_applied`), the score fields, the full config and the library versions, including the native
OpenSlide library.

`_patches.csv` has one row per tissue or edge tile: patch ID, role, `stored`, `h5_index` (row in the `.h5`,
blank if not stored), grid index, coordinates at 0.5 µm/px and full resolution, mask fraction, the scores
(mean RGB, grayscale median, mean saturation and brightness, Laplacian variance, and the near-white,
near-black, glass, dark and green fractions) and the pixel hash. Patch IDs
(`<source>_<16 hex of SHA-256(slide_id)>_<x>_<y>`), column names and shared score definitions match the
team's `gbm_patching.py`, so results can be compared at matched coordinates.

A slide with no tissue tiles is still written with zero tiles and logged as a warning, so it shows up
in the index with `n_tissue_tiles` 0 and is not retried on every run. The index and receipts give
`n_tissue_tiles` (all tissue tiles) and `n_stored_tiles` (tiles with pixels in the `.h5`).

Exit codes: `0` every selected slide tiled (or audited clean) · `1` slides failed, not yet downloaded,
or failed the audit · `2` setup or config problem · `130` interrupted with Ctrl-C.

## Tile QC decisions

`data_prep/qc.py` applies `configs/qc_policy.yaml` to the scores in each slide's `_patches.csv` and
gives every tissue and edge tile a decision, stored or not. Any reject reason outranks review. Edge tiles
(touching the mask, not more than half tissue) never pass automatically: the rules can reject them, and
the rest go to review with reason `edge`, so a reviewer can reinstate interpretable tissue with a patch
review. Nobody has to review every edge tile; unreviewed ones simply stay out of the model.

| Setting | Value | Meaning |
|---|---|---|
| `max_glass_fraction` | 0.9 | **Reject** a tile whose glass fraction is above 0.9. |
| `blank_fraction` | 0.99 | **Reject** a tile that is at least 99% near-white or near-black. |
| `blur_reject_below` | 50 | **Reject** a tile whose blur score (Laplacian variance) is below this: out of focus. |
| `blur_review_below` | 100 | **Review** a tile whose blur score is below this but not rejected. Between the two cutoffs, low scores can mean mild blur or sharp but smooth or pale tissue. With both cutoffs unset, a tile scoring exactly 0 (flat) still goes to review; a frozen policy must set both. |
| `dark_mean_rgb` | 50 | **Review** a tile whose mean R, G and B are all below 50. |
| `glass_review_above` | null | **Review** a tile whose glass fraction is above this (but not above `max_glass_fraction`). |
| `pale_review_below`, `pale_value_above` | null, 180 | **Review** a pale, washed-out tile: mean saturation below the first and mean brightness above the second. |
| `dark_fraction_review_above`, `green_fraction_review_above` | null, null | **Review** a tile with more dark pixels (folds, debris, ink) or green pixels (marker ink) than this. |
| `frozen`, `reviewer`, `notes` | false, "", "" | Set `frozen: true`, who approved the values and how they were chosen, once the team agrees them. `export.py` only builds a modeling list from a frozen policy. |
| `validation_fold` | null | Set the held-out TCGA fold (0–4). Required for a frozen policy. |
| `calibration_patients` | [] | TCGA patients whose tiles informed the cutoffs. For the chosen fold, each must be in the training partition. Required when frozen. |
| `require_signoff` | false | `true`: `export.py` blocks every slide without a sign-off (approve or exclude), and with it the slide's patient. |
| `stored_tiles_only` | true | Model only tissue tiles whose pixels are stored: the seeded random sample of up to `max_tiles_per_slide` per slide, the same on every run. Training then needs only the `.h5` files, not the slides, on any machine. Accepted tiles beyond the cap are counted (`accepted_not_stored`) but left out. The sample is random and independent of QC and labels, so it subsamples without biasing; it is a dataset decision recorded with the frozen policy. |

`null` turns an optional review rule off. Treat the blur cutoffs of 50 and 100 as candidates. Use
`review_qc.ipynb` to inspect accepted and flagged training tiles, including pale and low-texture
tissue, and to record MGMT-blind reviewer observations. Compare retention by slide and patient and
save the reviewed patients and threshold rationale in the policy. A missing, malformed, nonfinite,
or out-of-range QC score produces an `invalid_score` rejection. Re-audit the affected tile if needed.

Use a distinct policy version for each validation fold. Apply the same frozen policy to both A and
B for that fold and both model backbones. QC and export record the fold and fold-map hash. Existing
tiles and stored scores can be reused. Bump the policy version whenever the settings change.

```
python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B
```

It writes to `patches/<tiling version>/qc/<policy version>/`: `{source}/{patient_id}/{slide_id}_decisions.csv`
(patch ID, role, `stored`, `h5_index`, grid index, coordinates, decision, reasons; one row per tissue or
edge tile), `slides_<subset>.csv` (tissue decision counts and reasons per slide, edge counts separately),
`review_queue_<subset>.csv` (every tissue tile sent to review or rejected, review first, with patch ID,
coordinates and key scores: the reviewer's work list; edge tiles are left out of it) and
`policy_<subset>.json` (the policy, its hash and the tiling index it was applied to).

## Human review

`data_prep/review.py` keeps review records apart from the tiling outputs, as CSVs under
`<root>/review/<tiling version>/`. Use the notebook's "Recording reviews" cell or call it from Python:

```python
import review
review.append_patch_review(ROOT, "tiling-v7", patch_id, "reject", "Couger", "fold")
review.append_region(ROOT, "tiling-v7", "TCGA", slide_id, (x0, y0, x1, y1), "reject", "Couger", "pen")
review.sign_off_slide(ROOT, "tiling-v7", "TCGA", slide_id, "approve", "Couger", mask=True, focus=True)
review.sign_off_slide(ROOT, "tiling-v7", "TCGA", slide_id, "exclude", "Couger", notes="almost unstained")
```

- **Patch review:** accept, reject or review one tile, with a defect type. Accepting an edge tile reinstates it.
- **Artifact region:** a level-0 box around a technical artifact; every tissue tile it touches is rejected
  (or sent to review).
- **Slide sign-off:** approve a slide, recording what was checked, or exclude it with a reason. Sign-off is
  required for every slide only when the policy sets `require_signoff: true`; otherwise it records which
  slides a person reviewed (for example the flagged ones plus a random sample). An excluded slide counts as
  accounted for, so its patient is not blocked.

Records are only ever appended; the latest record for a tile or slide wins. `export.py` applies them on
top of the QC decision in this order: QC decision → artifact region → patch review → slide exclusion.
Defect types are technical damage only (fold, tear, bubble, pen, scanner stripe, ...): necrosis,
hemorrhage, microvascular proliferation and low cellularity are tissue, not defects. Keep MGMT labels out
of review.

## Export: the model input

`data_prep/export.py` builds `modeling_patches.csv`, the only list the model should read:

```
python data_prep/export.py --root ~/Bioinformatics/WSI/data --subset B
```

A tile is included only if:

- its slide was tiled with the current config, and the latest `--audit` covering it passed after tiling;
- the QC policy is frozen, its calibration patients are TCGA training patients for its validation fold, and `qc.py` applied it with the same fold map and tiling index;
- after reviews, its decision is accept (review cases and excluded slides stay out; an edge tile only if a
  reviewer accepted it);
- its patient is in the shared fold map (`manifests/patient_folds.csv`) and every one of the patient's
  slides in the subset is accounted for: exported, or excluded at sign-off (and, with `require_signoff`,
  every exported slide is approved at sign-off).

Accounting is per patient: a patient with a missing, failed or unaudited slide is listed as blocked, with
the reason, without blocking anyone else. A patient whose accepted tiles are all gone is listed as an
empty bag. Each modeling list serves its recorded validation fold. The loader checks the fold when assigning patient roles. With
`stored_tiles_only` (the default), accepted tiles beyond the tiling cap are left out and counted; edge tiles
a reviewer accepted are included (`stored` False until `package.py` stores them). The export is
**complete** when no patient is blocked; `export_settings.json` records it, and `package.py`, `combine.py`
and the loader refuse an incomplete export unless a partial one is asked for (a preview).

It writes `patches/<tiling version>/export/<policy version>/<subset>_<UTC time>/`: `modeling_patches.csv`
(one row per tile: patient, slide, fold, patch ID, coordinates, `stored`, `.h5` and row, slide path,
pixel hash, and what decided it), `patient_index.csv` (expected, exported and excluded slides, decision
counts, model tiles, reinstated edge tiles, empty bag, blocked reasons), `slide_status.csv` and
`export_settings.json` (the policy and the hashes of every input: review files, audits, the QC record, the
fold map and the modeling list, with paths relative to the data folder so they check on another machine).
Exit code `1` means some patients are blocked (the outputs are still written); `2` means a setup problem
such as an unfrozen policy.

An export is tied to its inputs: `loader.load_modeling` refuses it once any recorded input has changed or a
review or audit file has been added since (for example a new review record). Rerun `export.py` after
recording reviews.

`manifests/patient_folds.csv` is the shared fold map for all 234 patients: the 151 TCGA patients in five
folds (MGMT-stratified, seed 42; labels not stored here) and the 83 CPTAC patients marked `external`.
Both subsets and both backbones use the same map.

## Packages for training

Training runs on several machines, so it reads from a **package**: a portable copy of one export with
exactly its model tiles, built on the processing machine:

```
python data_prep/package.py --export <root>/patches/tiling-v7/export/qc-v1/B_<time> --out ~/packages/B_qc-v1
python data_prep/combine.py --parts A=~/packages/A_qc-v1 B=~/packages/B_qc-v1 --out ~/packages/AB_qc-v1
```

- `package.py` refuses a stale export (an input changed since) or an incomplete one (blocked patients;
  `--allow-partial` for a preview). It copies each model tile's PNG bytes unchanged into one `.h5` per slide
  under `tiles/`, reads any model tile that was not stored (an accepted edge tile) from its slide, checks
  every tile against its recorded pixel hash, and writes `package_manifest.json` with the SHA-256 and size
  of every file, the settings and the library versions. The output folder must be new.
- `combine.py` joins packages made with the same tiling config, code version, QC policy and fold map, with
  no patient in two parts. It is complete when every part is complete and together they account for all
  234 patients; otherwise it is refused unless `--allow-partial`. It copies no tiles: its rows point at
  each part's folder (relative paths), so keep the parts next to the combined folder, for example all in
  one `packages/` folder.

Store packages on OneDrive (or any shared drive) and copy them to local disk on the training machine:
reading thousands of random tiles per epoch through a sync client is slow. After copying, run
`loader.verify_package(FOLDER)` once to rehash every file. Training then needs only the package, h5py,
Pillow, NumPy and pandas: no slides and no OpenSlide.

## Reading tiles for training

`data_prep/loader.py` is for the training code; tiling never uses it.

```python
import loader
tiles = loader.with_roles(loader.load_modeling(FOLDER), fold=0)   # role: train, validation, external
reader = loader.TileReader(FOLDER)        # one per data-loader worker process
bag = loader.training_sample(tiles, patient_id, epoch)          # 64 tiles for this patient and epoch
images = reader.read_bag(bag)
for chunk in loader.evaluation_batches(tiles, patient_id, 64):  # every tile once, for evaluation
    images = reader.read_bag(chunk)

normalizer, record = loader.fit_fold_normalizer(FOLDER, 0, reference_patch_id, reader)
rgb = loader.normalize_tile(normalizer, rgb, patch_id, "logs/macenko_failures_fold0.csv")
```

`FOLDER` is a package or a combined folder (or, on the processing machine, an export folder). The
calls are the same for all three.

- **Training bags:** `training_sample` draws `count` tiles (default 64) for one training patient and epoch,
  round robin over the patient's slides and, within a slide, over 8 × 8-tile spatial blocks, so a bag
  covers the whole tissue rather than one region. The draw depends only on the seed, the epoch and the
  patient, so both backbones get the same tile IDs. A patient with fewer tiles gets all of them, then
  the rest again with replacement. It refuses a patient who is not a training patient for the fold.
- **Evaluation:** `evaluation_batches` yields every tile of a patient exactly once, in a fixed order.

`load_modeling` first checks the folder: for an export, that nothing it recorded has changed
(`loader.stale_inputs` lists what did); for a package or combination, the modeling list's hash and every
file's size (`loader.verify_package` rehashes everything); and that it is complete, unless
`allow_partial=True`. Every read checks the tile's pixel hash against the one recorded at tiling.
`TileReader` keeps at most 16 files open (least recently used). Each process should create its own
`TileReader`: HDF5 and slide handles are not shared across processes.

Macenko stain normalization happens here, at training time, not in tiling: its reference has to come from
each fold's own training patients, and the cache stays a single set of raw tiles. `fit_fold_normalizer`
only accepts a reference tile from a TCGA training patient of that fold in the exported list; save the
returned `record` with the model run. Normalize each tile before augmentation and tensor normalization,
with the same fold reference for training, validation and CPTAC tiles. `normalize_tile` logs a failed
transform and raises: a raw tile is never passed on as if it had been normalized.

## Tests

`tests/run_tests.py` checks the whole pipeline end to end on small synthetic slides
(`tests/make_slides.py` writes fake Aperio `.svs` files; no study data is used):

```
python tests/run_tests.py           # in the gbm-mgmt environment, from the repo folder
python tests/run_tests.py --keep    # keep the temporary data folder for inspection
```

It covers registering copies, tiling, the resolution limit, the mask floor, lossless storage, the cap,
rereading tiles, the audit (rereads of unstored tiles, slide rehashing, a changed pixel or QC image caught),
the config and `--slides` guards, QC rules (blur tiers, edge tiles, frozen-policy requirements, the review
queue), reviews (including reinstating an edge tile), the export gates and accounting (calibration
patients, required sign-off, stored tiles only, completeness), stale export detection, packages (an edge
tile reread from its slide, a stale or incomplete export refused, a damaged file caught), combining
halves A and B, the patient bag sampler, the loader, Macenko, and `run_pipeline.sh`. Set `GBM_PATCHING_DIR` to a folder with
the team's `gbm_patching.py` to also check that both tilers read identical pixels.

## Repository layout

```
data_prep/download.py   download with resume and verification; --register for copied slides
data_prep/split.py      patient-level A/B work split
data_prep/common.py     shared helpers: config checks, paths, patch IDs, fold map
data_prep/slides.py     slide reading, tissue mask, tile grid, scores (TIAToolbox)
data_prep/tile.py       tile extraction, mask comparison, integrity audit
data_prep/qc.py         apply the tile QC policy
data_prep/review.py     human review records
data_prep/export.py     modeling list with per-patient accounting
data_prep/package.py    portable copy of an export with its model tiles
data_prep/combine.py    join packages (halves A and B)
data_prep/loader.py     read tiles, patient bags and Macenko per fold, for training code
scripts/run_pipeline.sh run the steps in order for one subset
configs/tiling.yaml     tiling settings
configs/qc_policy.yaml  tile QC cutoffs
manifests/              frozen slide manifest, shared patient fold map
tests/                  end-to-end tests on synthetic slides
logs/                   run logs (ignored by git; add submitted logs with git add -f)
environment.yml         conda environment
requirements.txt        pip requirements (also used by environment.yml)
```

## Status

- [x] Slide download and verification
- [ ] Tissue masking and patch extraction (pipeline built; mask review and full run pending)
- [ ] UNI2-h embeddings and PixCell synthetic bags
- [ ] Model training (ConvNeXt V2-Tiny, Swin V2-Tiny) and evaluation
