"""Code for review_qc.ipynb: open a review session, sample and show tiles, and record decisions.

The notebook keeps the settings and calls these functions. Production records (patch decisions and
slide sign-off) go through data_prep/review.py, so export.py applies them; the calibration journal is
a separate record of what was viewed while choosing cutoffs.
"""
import csv
import html
import io
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


def repo_root(start=None):
    """The repository folder: `start` (default: the working folder) or its nearest parent with
    configs/ and data_prep/, so the notebook works from notebooks/ or from the repository root."""
    start = Path(start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "configs").is_dir() and (folder / "data_prep").is_dir():
            return folder
    raise FileNotFoundError(f"no repository folder above {start}")


REPO = repo_root(Path(__file__).parent)
if str(REPO / "data_prep") not in sys.path:
    sys.path.insert(0, str(REPO / "data_prep"))

import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.colors import to_rgb                # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

import common                                       # noqa: E402
import grandqc                                      # noqa: E402
import loader                                       # noqa: E402
import qc                                           # noqa: E402
import review                                       # noqa: E402

JOURNAL_FIELDS = ["patient_id", "slide_id", "patch_id", "decision", "automatic_reasons",
                  "usable", "confirmed_defect", "reviewer", "notes"]
USABLE = ("usable", "unusable", "uncertain")
TILE_FIELDS = ("patch_id", "patient_id", "slide_id", "role", "stored", "h5_index", "slide_relpath", "x_target",
               "y_target", "x0", "y0", "x1", "y1", "laplacian_variance", "pixel_sha256")   # what pages keep
PAGED_LIMIT = 50_000      # most tiles one selection keeps for paging (a seeded random subset past this)
TABLE_COLUMNS = ["patch_id", "slide_id", "x0", "y0", "decision", "automatic_reasons",
                 "laplacian_variance", "grandqc", "confirmed_defect", "stored"]


@dataclass
class Session:
    """What the notebook's cells share: the data, the policy and the cohort's slides."""
    root: Path
    subset: str
    cohort: str
    policy: dict
    policy_hash: str
    tiling: Path
    tiling_version: str
    folds: dict
    fold_hash: str
    slides: list            # the cohort's index rows in the subset
    patients: list
    journal: Path           # calibration journal, scoped to the policy and fold map


def skip_scale_warning(record):
    """Drop TIAToolbox's "Read: Scale > 1" warning, logged on every re-read from a slide scanned
    slightly coarser than 0.5 um/px (the read is upsampled by under 1%)."""
    return not str(record.msg).startswith("Read: Scale > 1")


def open_session(root, subset, policy_path, cohort="TCGA", tiling=None, fold_map=None):
    """Load and check the policy and fold map, and find the cohort's slides in the subset."""
    root = Path(root)
    tiling = Path(tiling or REPO / "configs" / "tiling.yaml")
    fold_map = Path(fold_map or REPO / "manifests" / "patient_folds.csv")
    root_logger = logging.getLogger()      # TIAToolbox logs to the root logger
    if not any(getattr(f, "__name__", "") == "skip_scale_warning" for f in root_logger.filters):
        root_logger.addFilter(skip_scale_warning)

    policy, policy_hash, _ = qc.load_policy(policy_path)
    if cohort not in policy["sources"]:
        raise ValueError(f"{cohort} is not covered by policy {policy['version']}: {policy['sources']}")
    fold_hash = common.sha256_file(fold_map)
    folds = common.read_folds(fold_map)
    if common.sha256_file(fold_map) != fold_hash:
        raise RuntimeError("Fold map changed while reading.")
    eligible = qc.calibration_patients(folds, [cohort])
    qc.validate_calibration(policy, folds)
    version = common.tiling_version(tiling)
    index_path = root / "patches" / version / f"index_{subset}.csv"
    slides = [row for row in common.read_rows(index_path)
              if row["source"] == cohort and row["patient_id"] in eligible]
    patients = sorted({row["patient_id"] for row in slides})
    if not patients:
        raise ValueError(f"No {cohort} patients in {index_path}")
    journal = root / "review" / version / "calibration" / f"{policy_hash}_{fold_hash}.csv"
    return Session(root, subset, cohort, policy, policy_hash, tiling, version, folds, fold_hash,
                   slides, patients, journal)


def slide_summary(session):
    """The cohort's rows of qc.py's slides_<subset>.csv for this policy (None when it is missing)."""
    path = (session.root / "patches" / session.tiling_version / "qc" / session.policy["version"]
            / f"slides_{session.subset}.csv")
    if not path.exists():
        return None
    summary = pd.read_csv(path, dtype=str, keep_default_na=False)
    return summary[summary["source"] == session.cohort]


def flagged_slides(session):
    """The cohort's slides that qc.py flagged for a sign-off decision, with tissue and edge counts
    (None when qc.py has not written slides_<subset>.csv for this policy)."""
    summary = slide_summary(session)
    if summary is None:
        return None
    flagged = summary[summary["slide_flags"] != ""]
    return (flagged[["patient_id", "slide_id", "slide_flags", "accept", "review", "reject", "edge_accept",
                     "edge_review", "edge_reject", "median_mean_saturation"]]
            .rename(columns={"accept": "tissue_accept", "review": "tissue_review", "reject": "tissue_reject"})
            .reset_index(drop=True))


def pen_slides(session, candidate_above=0.01):
    """GrandQC pen candidates for the cohort: slides with a GrandQC pen fraction above `candidate_above`,
    plus every slide in the policy's pen_slides, largest pen fraction first. `listed` marks the slides on
    the list (only their pen tiles go to review), and `gq_pen` counts those tissue tiles. Empty when the
    pen rule is off; None when qc.py has not written slides_<subset>.csv for this policy."""
    summary = slide_summary(session)
    if summary is None:
        return None
    columns = ["patient_id", "slide_id", "listed", "pen_fraction", "gq_pen", "tissue_tiles", "accept", "review",
               "reject"]
    if session.policy["pen_tile_review_above"] is None or not session.policy["grandqc"]:
        return pd.DataFrame(columns=columns)
    listed = set(session.policy["pen_slides"])
    fraction = pd.to_numeric(summary["pen_fraction"], errors="coerce")
    pen = summary[(fraction > candidate_above) | summary["slide_id"].isin(listed)].copy()
    pen["pen_fraction"] = pd.to_numeric(pen["pen_fraction"], errors="coerce").fillna(0.0)
    pen["listed"] = pen["slide_id"].isin(listed)
    pen["gq_pen"] = [json.loads(r or "{}").get("gq_pen", 0) for r in pen["reasons"]]
    return (pen.sort_values("pen_fraction", ascending=False)[columns]
            .rename(columns={"accept": "tissue_accept", "review": "tissue_review", "reject": "tissue_reject"})
            .reset_index(drop=True))


def grandqc_missing(session):
    """The cohort's slides whose GrandQC data is not current (export blocks their patients), or None
    when qc.py has not written slides_<subset>.csv for this policy."""
    summary = slide_summary(session)
    if summary is None:
        return None
    return summary[summary["grandqc"] != "ok"][["patient_id", "slide_id", "grandqc"]].reset_index(drop=True)


def show_pen_overview(session, pen, columns=3, review_color="orange", accepted_color="cyan"):
    """Each slide in `pen` (from pen_slides) as its QC thumbnail with its tiles shaded: `review_color`
    for tiles sent to review for pen (gq_pen; listed slides only), `accepted_color` for accepted tiles
    where GrandQC still found pen (at least 0.01 of the tile). Colors are any matplotlib color name or hex
    string."""
    slide_rows = {s["slide_id"]: s for s in session.slides}
    chosen = [slide_rows[slide_id] for slide_id in pen["slide_id"] if slide_id in slide_rows]
    if not chosen:
        print("No pen slides to show.")
        return
    columns = min(columns, len(chosen))
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    review_rgb, accepted_rgb = np.array(to_rgb(review_color)), np.array(to_rgb(accepted_color))
    for ax, slide in zip(axes.flat, chosen):
        thumb = plt.imread(common.output_paths(session.root, slide, session.tiling_version)[1])
        image = (thumb[..., :3].astype(float) / (255 if thumb.dtype == np.uint8 else 1)).copy()
        scale = image.shape[1] / grandqc.slide_geometry(session.root / slide["h5"])[0]   # per level-0 px
        gq_tiles = grandqc_tiles(session, slide)
        counts = {"gq_pen": 0, "accepted with pen": 0}
        with (session.root / slide["csv"]).open(newline="") as handle:
            for patch in csv.DictReader(handle):
                if patch["role"] not in qc.ROLES:
                    continue
                gq = gq_tiles.get(patch["patch_id"])
                if gq is None:
                    continue
                decision, reasons = qc.decide(patch, session.policy, gq)
                if "gq_pen" in [r.split(":", 1)[0] for r in reasons.split(";")]:
                    color, key = review_rgb, "gq_pen"
                elif decision == "accept" and gq["pen"] >= 0.01:
                    color, key = accepted_rgb, "accepted with pen"
                else:
                    continue
                counts[key] += 1
                x0, y0 = int(int(patch["x0"]) * scale), int(int(patch["y0"]) * scale)
                x1 = max(int(int(patch["x1"]) * scale), x0 + 1)
                y1 = max(int(int(patch["y1"]) * scale), y0 + 1)
                image[y0:y1, x0:x1] = 0.45 * image[y0:y1, x0:x1] + 0.55 * color
        ax.imshow(image)
        row = pen.loc[pen["slide_id"] == slide["slide_id"]].iloc[0]
        status = "LISTED" if row.get("listed", False) else "not listed"
        ax.set_title(f"{slide['patient_id']}  {slide['slide_id'][:8]}  pen {row['pen_fraction']:.3f}  {status}\n"
                     f"{review_color} gq_pen {counts['gq_pen']:,}  |  "
                     f"{accepted_color} accepted with pen {counts['accepted with pen']:,}",
                     fontsize=9)
    fig.tight_layout()
    plt.show()


STAT_COUNTS = ["tissue_tiles", "tissue_accept", "tissue_review", "tissue_reject",
               "edge_tiles", "edge_accept", "edge_review", "edge_reject", "all_tiles"]


def slide_stats(session, cap=None):
    """Per-slide tile counts from qc.py's slides_<subset>.csv (QC decisions, before human review), with
    `est_export`: accepted tissue + edge tiles up to the per-slide cap (the tiling config's
    max_tiles_per_slide unless `cap` is given), 0 for a slide excluded at sign-off. Patch decisions and
    region boxes are not counted, so export's own receipt is the final number. None without the CSV."""
    summary = slide_summary(session)
    if summary is None:
        return None
    if cap is None:
        cap = common.load_tiling_config(session.tiling)[0]["max_tiles_per_slide"]
    signoffs = review.load_reviews(session.root, session.tiling_version).signoffs
    listed = set(session.policy["pen_slides"]) if session.policy["grandqc"] else set()
    rows = []
    for r in summary.to_dict("records"):
        counts = {k: int(r[k] or 0) for k in ("tissue_tiles", "accept", "review", "reject", "edge_tiles",
                                               "edge_accept", "edge_review", "edge_reject")}
        signed = signoffs.get((session.cohort, r["slide_id"]), {}).get("decision", "")
        accepted = counts["accept"] + counts["edge_accept"]
        export = 0 if signed == "exclude" else (accepted if cap is None else min(accepted, cap))
        flags = [f for f in r["slide_flags"].split(";") if f]
        if r["slide_id"] in listed:
            flags.append("pen")
        rows.append(dict(patient_id=r["patient_id"], slide_id=r["slide_id"],
                         tissue_tiles=counts["tissue_tiles"], tissue_accept=counts["accept"],
                         tissue_review=counts["review"], tissue_reject=counts["reject"],
                         edge_tiles=counts["edge_tiles"], edge_accept=counts["edge_accept"],
                         edge_review=counts["edge_review"], edge_reject=counts["edge_reject"],
                         all_tiles=counts["tissue_tiles"] + counts["edge_tiles"],
                         accept_pct=round(100 * counts["accept"] / counts["tissue_tiles"], 1)
                         if counts["tissue_tiles"] else 0.0,
                         est_export=export, below_cap=signed != "exclude" and cap is not None and accepted < cap,
                         flags=";".join(flags), signed_off=signed))
    return (pd.DataFrame(rows).sort_values(["est_export", "patient_id", "slide_id"]).reset_index(drop=True))


def patient_stats(session, cap=None):
    """Per-patient totals of slide_stats, fewest estimated export tiles first. `flags` lists flagged
    slides as <slide ID start>:<flags>; `signed_off` counts approvals and exclusions. None without the CSV."""
    slides = slide_stats(session, cap)
    if slides is None:
        return None
    rows = []
    for patient, group in slides.groupby("patient_id"):
        tissue = int(group["tissue_tiles"].sum())
        decisions = group["signed_off"][group["signed_off"] != ""].value_counts()
        rows.append(dict(patient_id=patient, slides=len(group), **{k: int(group[k].sum()) for k in STAT_COUNTS},
                         accept_pct=round(100 * int(group["tissue_accept"].sum()) / tissue, 1) if tissue else 0.0,
                         est_export=int(group["est_export"].sum()), slides_below_cap=int(group["below_cap"].sum()),
                         flags=" ".join(f"{sid[:8]}:{f}" for sid, f in zip(group["slide_id"], group["flags"]) if f),
                         signed_off=", ".join(f"{n} {d}" for d, n in sorted(decisions.items()))))
    return pd.DataFrame(rows).sort_values(["est_export", "patient_id"]).reset_index(drop=True)


def select_slides(session, patient=None, slide=None):
    """The slides to sample: one slide by the start of its ID (overrides `patient`), one patient's
    slides, or with `patient` None every slide of the cohort (TCGA only)."""
    if slide is not None:
        chosen = [s for s in session.slides if s["slide_id"].startswith(slide)]
        if len(chosen) != 1:
            raise ValueError(f"SLIDE {slide!r} matches {len(chosen)} {session.cohort} slides in this subset.")
        return chosen
    if patient is None:
        if session.cohort != "TCGA":
            raise ValueError("PATIENT = None views every patient: TCGA only.")
        print(f"Reading all {len(session.slides)} {session.cohort} slides...")
        return list(session.slides)
    if patient not in session.patients:
        raise ValueError(f"Choose a {session.cohort} patient in this subset.")
    return [s for s in session.slides if s["patient_id"] == patient]


def grandqc_tiles(session, slide):
    """The slide's GrandQC fractions per tile when the policy uses them ({} when off or missing)."""
    if not session.policy["grandqc"]:
        return {}
    tiles, _, why = qc.grandqc_context(session.root, session.tiling_version, slide, session.policy,
                                       qc.tissue_saturation(session.root / slide["csv"]))
    if tiles is None:
        print(f"{slide['slide_id']}: {why}; GrandQC rules skipped here (run grandqc.py)")
        return {}
    return tiles


def short_grandqc(gq):
    """GrandQC fractions worth showing: assessed when below 1, and artifacts of at least 0.01."""
    if gq is None:
        return ""
    parts = [] if gq["assessed"] >= 0.995 else [f"assessed {gq['assessed']:.2f}"]
    parts += [f"{k} {gq[k]:.2f}" for k in ("oof", "pen", "fold", "darkspot", "bubble") if gq[k] >= 0.01]
    return "gq " + (" ".join(parts) if parts else "clean")


def match_tiles(session, slides, decision=None, reason=None, confirmed_defect=None, blur_band=None,
                role=None, seed=42, limit=PAGED_LIMIT):
    """Every tile in `slides` that passes every filter (None = any value), in a seeded random order.
    Returns (tiles, number of matching tiles); past `limit` matches, `tiles` is a seeded random `limit`
    of them. Each tile keeps only TILE_FIELDS, its QC decision and reasons, and a short GrandQC line."""
    if role not in (None, *qc.ROLES):
        raise ValueError(f"ROLE must be None or one of {qc.ROLES}.")
    if limit < 1:
        raise ValueError("limit must be positive.")
    previous = ({row["patch_id"]: row for row in common.read_rows(session.journal)}
                if session.journal.exists() else {})
    rng = random.Random(seed)
    tiles, matches = [], 0
    for slide in slides:
        gq_tiles = grandqc_tiles(session, slide)
        with (session.root / slide["csv"]).open(newline="") as handle:
            for patch in csv.DictReader(handle):
                if patch["role"] not in qc.ROLES or (role is not None and patch["role"] != role):
                    continue
                gq = gq_tiles.get(patch["patch_id"])
                tile_decision, reasons = qc.decide(patch, session.policy, gq)
                tags = [r.split(":", 1)[0] for r in reasons.split(";")]
                confirmed = previous.get(patch["patch_id"], {}).get("confirmed_defect", "")
                if decision is not None and tile_decision != decision:
                    continue
                if reason is not None and reason not in tags:
                    continue
                if confirmed_defect is not None and confirmed != confirmed_defect:
                    continue
                if blur_band is not None:
                    focus = qc.read_scores(patch)[0].get("laplacian_variance")
                    if focus is None or not blur_band[0] <= focus < blur_band[1]:
                        continue
                matches += 1
                slot = rng.randrange(matches) if len(tiles) == limit else len(tiles)
                if slot < limit:
                    item = {k: patch.get(k, "") for k in TILE_FIELDS}
                    item.update(h5=slide["h5"], decision=tile_decision, automatic_reasons=reasons,
                                confirmed_defect=confirmed, grandqc=short_grandqc(gq))
                    if slot == len(tiles):
                        tiles.append(item)
                    else:
                        tiles[slot] = item
    rng.shuffle(tiles)
    return tiles, matches


class TilePager:
    """Pages through matched tiles `per_page` at a time. Montage numbers restart at 0 on each page, and
    current() is the page on show, for the record cells. show() adds Previous / Next buttons."""

    def __init__(self, session, tiles, per_page=12, page=0, thumbnails=True, columns=4, tile_size=3.5):
        if per_page < 1:
            raise ValueError("N must be positive.")
        if columns < 1 or tile_size <= 0:
            raise ValueError("COLUMNS must be at least 1 and TILE_SIZE positive.")
        self.session, self.tiles, self.per_page, self.thumbnails = session, tiles, per_page, thumbnails
        self.columns, self.tile_size = columns, tile_size
        self.pages = max(1, math.ceil(len(tiles) / per_page))
        self.page = 0
        self.go(page)
        self.seen = set()          # patients on the pages shown so far

    def go(self, page):
        """Move to `page` (0-based), kept within the first and last page."""
        self.page = min(max(int(page), 0), self.pages - 1)

    def current(self):
        """The tiles on the page on show."""
        start = self.page * self.per_page
        return self.tiles[start:start + self.per_page]

    def viewed_patients(self):
        """Patients on any page shown: each belongs in the policy's calibration_patients."""
        return sorted(self.seen)

    def header(self):
        """Page line and the patients viewed so far (marks the current page's patients as viewed)."""
        self.seen.update(patch["patient_id"] for patch in self.current())
        return (f"Page {self.page + 1} of {self.pages}  ({len(self.tiles):,} tiles)",
                f"Patients viewed (calibration_patients): {self.viewed_patients()}")

    def render(self):
        """Show the current page (montage, table, thumbnails) as ordinary cell output."""
        from IPython.display import display
        for line in self.header():
            print(line)
        sample = self.current()
        if not sample:
            print("No matches. Try clearing a filter or choosing another patient.")
            return
        show_montage(self.session, sample, self.columns, self.tile_size)
        display(sample_table(sample))
        if self.thumbnails:
            show_thumbnails(self.session, sample)

    def show(self):
        """Previous / Next buttons above the current page; without ipywidgets, the current page only.
        Each page is drawn into image and HTML widgets (not captured output), so it replaces the last
        page in every notebook front end."""
        try:
            import ipywidgets as widgets
        except ImportError:
            print("ipywidgets is not installed (pip install ipywidgets): change PAGE and re-run to page.")
            self.render()
            return
        from IPython.display import display
        previous_button = widgets.Button(description="◀ Previous")
        next_button = widgets.Button(description="Next ▶")
        info = widgets.HTML()
        montage = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
        table = widgets.HTML()
        thumbnails = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))

        def draw():
            previous_button.disabled = self.page == 0
            next_button.disabled = self.page >= self.pages - 1
            lines = list(self.header())
            sample = self.current()
            try:
                if sample:
                    fig, errors = montage_figure(self.session, sample, self.columns, self.tile_size)
                    montage.value = figure_png(fig)
                    lines += errors
                    table.value = sample_table(sample).to_html(index=True)
                    thumbnails.value = figure_png(thumbnails_figure(self.session, sample)) if self.thumbnails else b""
                else:
                    lines.append("No matches. Try clearing a filter or choosing another patient.")
                    montage.value, table.value, thumbnails.value = b"", "", b""
            except Exception as exc:          # a callback's error is otherwise lost
                lines.append(f"Error drawing the page: {type(exc).__name__}: {exc}")
            info.value = "<br>".join(html.escape(line) for line in lines)

        def step(delta):
            self.go(self.page + delta)
            draw()

        previous_button.on_click(lambda _: step(-1))
        next_button.on_click(lambda _: step(1))
        draw()
        display(widgets.VBox([widgets.HBox([previous_button, next_button]), info, montage, table, thumbnails]))


def tile_title(number, patch, compact=False):
    """Title lines: decision and focus; slide, position and cache; reasons; GrandQC. `compact` (for small
    tiles) keeps number, decision and focus, the reasons, and the GrandQC line only when not clean."""
    focus = patch.get("laplacian_variance", "")
    focus = f"{float(focus):.0f}" if focus not in ("", None) else "missing"
    reasons = patch["automatic_reasons"] + (f" | {patch['confirmed_defect']}" if patch["confirmed_defect"] else "")
    if compact:
        title = f"{number}  {patch['decision']}  {focus}\n{reasons}"
        return title + (f"\n{patch['grandqc']}" if patch["grandqc"] not in ("", "gq clean") else "")
    source = "cached" if str(patch["stored"]) == "True" else "re-read"
    title = (f"{number}  {patch['decision']}  focus {focus}\n"
             f"{patch['slide_id'][:8]} @ {patch['x0']},{patch['y0']}  {source}\n{reasons}")
    return title + (f"\n{patch['grandqc']}" if patch["grandqc"] else "")


COMPACT_BELOW = 3.0      # tile size (inches) under which titles are shortened


def montage_figure(session, sample, columns=4, tile_size=3.5):
    """(figure of the sampled tiles, list of read errors): `columns` per row at `tile_size` inches,
    numbered as in the table and the thumbnails. Below COMPACT_BELOW inches the titles are shortened."""
    if columns < 1 or tile_size <= 0:
        raise ValueError("COLUMNS must be at least 1 and TILE_SIZE positive.")
    compact = tile_size < COMPACT_BELOW
    rows = (len(sample) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * tile_size, rows * (tile_size + (0.5 if compact else 0.2))),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    errors = []
    with loader.TileReader(session.root, tiling_config=session.tiling) as reader:
        for number, (ax, patch) in enumerate(zip(axes.flat, sample)):
            try:
                ax.imshow(reader.read(patch))
            except (RuntimeError, OSError, ValueError, KeyError) as exc:
                ax.text(0.5, 0.5, "Read failed", ha="center")
                errors.append(f"{number}: {patch['patch_id']} {exc}")
            ax.set_title(tile_title(number, patch, compact), fontsize=7 if compact else 8.5)
    fig.tight_layout()
    return fig, errors


def show_figure(fig):
    """Display a figure once in the current output and close it (so it is not shown a second time)."""
    from IPython.display import display
    display(fig)
    plt.close(fig)


def figure_png(fig):
    """A figure as PNG bytes (for a widget), closing it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def show_montage(session, sample, columns=4, tile_size=3.5):
    """Display the montage of the sampled tiles, then any read errors."""
    fig, errors = montage_figure(session, sample, columns, tile_size)
    show_figure(fig)
    for error in errors:
        print(error)


def sample_table(sample):
    """The sampled tiles as a table, one row per montage number."""
    return pd.DataFrame(sample).reindex(columns=TABLE_COLUMNS)


def thumbnails_figure(session, sample):
    """Figure of each sampled slide's QC thumbnail (green: tissue mask, blue: some stored tiles), with
    the sampled tiles boxed and numbered in orange."""
    by_slide = {}
    for number, patch in enumerate(sample):
        by_slide.setdefault(patch["slide_id"], []).append((number, patch))
    slide_rows = {s["slide_id"]: s for s in session.slides}
    columns = min(3, len(by_slide))
    rows = (len(by_slide) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (slide_id, numbered) in zip(axes.flat, by_slide.items()):
        slide = slide_rows[slide_id]
        thumb = plt.imread(common.output_paths(session.root, slide, session.tiling_version)[1])
        scale = thumb.shape[1] / grandqc.slide_geometry(session.root / slide["h5"])[0]   # per level-0 px
        ax.imshow(thumb)
        for number, patch in numbered:
            x, y = int(patch["x0"]) * scale, int(patch["y0"]) * scale
            size = max((int(patch["x1"]) - int(patch["x0"])) * scale, 6)
            ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="orange", linewidth=1.5))
            ax.annotate(str(number), (x + size, y), color="orange", fontsize=9, fontweight="bold")
        ax.set_title(f"{slide['patient_id']}  {slide_id[:8]}", fontsize=9)
    fig.tight_layout()
    return fig


def show_thumbnails(session, sample):
    """Display the sampled slides' thumbnails with the sampled tiles numbered."""
    show_figure(thumbnails_figure(session, sample))


def save_observation(session, sample, row, usable, confirmed, reviewer, notes=""):
    """Add one montage tile to the calibration journal (does not change the export)."""
    if not 0 <= row < len(sample):
        raise ValueError("Choose an image shown in the current montage.")
    if not reviewer.strip() or usable not in USABLE or confirmed not in review.DEFECTS:
        raise ValueError("Enter a reviewer, usability decision and valid defect type.")
    patch = sample[row]
    review._append(session.journal, JOURNAL_FIELDS, {
        "patient_id": patch["patient_id"], "slide_id": patch["slide_id"], "patch_id": patch["patch_id"],
        "decision": patch["decision"], "automatic_reasons": patch["automatic_reasons"],
        "usable": usable, "confirmed_defect": confirmed, "reviewer": reviewer.strip(), "notes": notes,
    })
    print("Saved", patch["patch_id"])


def observation_summary(session):
    """(reviewed patient IDs, counts by usability and defect) from the journal, or None if empty."""
    if not session.journal.exists():
        return None
    observations = pd.read_csv(session.journal).drop_duplicates("patch_id", keep="last")
    return (sorted(observations["patient_id"].unique()),
            observations.groupby(["usable", "confirmed_defect"]).size().rename("patches"))


def record_decisions(session, sample, rows, record, defect, sign_off=None, checks=None, reviewer="", notes=""):
    """Record patch decisions for montage tiles `rows` and, optionally, sign off the one slide shown.
    Every input is checked before anything is written."""
    checks = checks or {}
    slides_shown = sorted({patch["slide_id"] for patch in sample})
    if not reviewer.strip():
        raise ValueError("Enter a reviewer.")
    if any(not 0 <= row < len(sample) for row in rows):
        raise ValueError("Choose images shown in the current montage.")
    if record not in ("accept", "reject", "review") or defect not in review.DEFECTS:
        raise ValueError(f"bad decision: {record!r}, {defect!r}")
    if sign_off is not None:
        if sign_off not in ("approve", "exclude") or (sign_off == "exclude" and not notes.strip()):
            raise ValueError(f"bad sign-off: {sign_off!r}, notes {notes!r}")
        if len(slides_shown) != 1:
            raise ValueError(f"montage shows {len(slides_shown)} slides; set SLIDE")
        if set(checks) - set(review.CHECKS) or any(not isinstance(v, bool) for v in checks.values()):
            raise ValueError(f"checks must be True/False for {review.CHECKS}: {checks}")
    for row in rows:
        patch = sample[row]
        review.append_patch_review(session.root, session.tiling_version, patch["patch_id"], record,
                                   reviewer, defect, notes)
        print(f"{row}: {record} {patch['patch_id']}")
    if sign_off is not None:
        review.sign_off_slide(session.root, session.tiling_version, session.cohort, slides_shown[0], sign_off,
                              reviewer, notes, **checks)
        print(f"{slides_shown[0]}: {sign_off}")


def decision_status(session, sample):
    """Print and return the sign-off of each slide in the montage, and the montage tiles that have a
    patch decision: ({slide_id: decision or "none"}, {montage number: decision})."""
    reviews = review.load_reviews(session.root, session.tiling_version)
    signoffs = {slide_id: reviews.signoffs.get((session.cohort, slide_id), {}).get("decision") or "none"
                for slide_id in sorted({patch["slide_id"] for patch in sample})}
    recorded = {row: reviews.patches[patch["patch_id"]]["decision"]
                for row, patch in enumerate(sample) if patch["patch_id"] in reviews.patches}
    for slide_id, decision in signoffs.items():
        print(f"{slide_id[:8]}: sign-off {decision}")
    print("Montage tiles with a patch decision:", recorded or "none")
    return signoffs, recorded


def sweep_blur(session, patients, candidates):
    """Tissue-tile decision counts under each (reject below, review below) blur pair, other rules
    fixed: (per-slide table, per-patient table), or (None, None) when no slide is selected."""
    if not set(patients) <= set(session.patients):
        raise ValueError(f"Sweep patients must be {session.cohort} patients in this subset.")
    if any(not 0 <= reject or (check is not None and check < reject) for reject, check in candidates):
        raise ValueError("Each pair must satisfy 0 <= reject cutoff <= review cutoff (or review None).")
    trials = [dict(session.policy, blur_reject_below=reject, blur_review_below=check)
              for reject, check in candidates]
    results = []
    for slide in session.slides:
        if slide["patient_id"] not in patients:
            continue
        counts = [{"accept": 0, "review": 0, "reject": 0, "accepted_cached": 0} for _ in candidates]
        gq_tiles = grandqc_tiles(session, slide)
        with (session.root / slide["csv"]).open(newline="") as handle:
            for patch in csv.DictReader(handle):
                if patch["role"] != "tissue":
                    continue
                for trial, tally in zip(trials, counts):
                    decision, _ = qc.decide(patch, trial, gq_tiles.get(patch["patch_id"]))
                    tally[decision] += 1
                    if decision == "accept" and patch["stored"] == "True":
                        tally["accepted_cached"] += 1
        for (reject, check), tally in zip(candidates, counts):
            results.append(dict(patient_id=slide["patient_id"], slide_id=slide["slide_id"],
                                reject_below=reject, review_below=check, **tally))
    if not results:
        return None, None
    slide_counts = pd.DataFrame(results)
    columns = ["accept", "review", "reject", "accepted_cached"]
    patient_counts = slide_counts.groupby(["patient_id", "reject_below", "review_below"], as_index=False,
                                          dropna=False)[columns].sum()
    patient_counts["accepted_fraction"] = patient_counts["accept"].div(
        patient_counts[["accept", "review", "reject"]].sum(axis=1))
    return slide_counts, patient_counts
