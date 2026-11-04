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

## Repository layout

```
data_prep/download.py   download with resume and verification
data_prep/split.py      patient-level A/B split
manifests/              frozen slide manifest
logs/                   run logs (ignored by git; add submitted logs with git add -f)
environment.yml         conda environment
requirements.txt        pip requirements (Colab)
```

## Status

- [x] Slide download and verification
- [ ] Tissue masking and patch extraction
- [ ] UNI2-h embeddings and PixCell synthetic bags
- [ ] Model training (ConvNeXt V2-Tiny, Swin V2-Tiny) and evaluation
