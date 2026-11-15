"""End-to-end tests of the tiling pipeline on small synthetic slides. Uses no study data.

    python tests/run_tests.py            # from the repo folder, in the gbm-mgmt environment
    python tests/run_tests.py --keep     # keep the temporary data folder for inspection

Builds a temporary data folder with four synthetic .svs files (tests/make_slides.py) placed at the
paths of five real manifest rows, with a test copy of the frozen manifest, then runs
download.py --register, tile.py (tiling, cap, --masks-only, --audit), qc.py, review.py, export.py,
loader.py (including Macenko) and scripts/run_pipeline.sh, and checks the results.

If GBM_PATCHING_DIR points to a folder with the team's gbm_patching.py, it also checks that both
tilers read identical pixels at matched coordinates; otherwise that check is skipped.

Exit code 0 = every check passed, 1 = some failed.
"""
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "data_prep"))
sys.path.insert(0, str(REPO / "tests"))


def main():
    import h5py
    import numpy as np
    from PIL import Image

    import common
    import download
    import export
    import loader
    import combine
    import make_slides
    import package
    import qc
    import review
    import slides
    import tile

    keep = "--keep" in sys.argv
    work = Path(tempfile.mkdtemp(prefix="gbm-tests-"))
    root, logs = work / "data", work / "logs"
    results = {}

    def check(name, ok, detail=""):
        results[name] = bool(ok)
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)

    def read_csv(path):
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle))

    # ------------------------------------------------------------------ setup: slides and manifest
    print(f"Test data folder: {work}")
    made = make_slides.make_all(work / "slides")
    manifest = json.loads((REPO / "manifests" / "collection_manifest.json").read_text())
    files = manifest["files"]
    tcga = [r for r in files if r["source"] == "TCGA"]
    cptac = [r for r in files if r["source"] == "CPTAC"]
    picks = {"mixed": tcga[0], "faint": tcga[1], "corrupt": tcga[2], "coarse_ok": cptac[0],
             "coarse_bad": cptac[1]}
    def place(name, row):
        dest = root / "raw" / row["source"] / row["patient_id"] / row["file_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(made["mixed" if name in ("corrupt", "single_b") else name], dest)
        row["size_bytes"] = dest.stat().st_size
        if row["source"] == "TCGA":
            row["md5"] = hashlib.md5(dest.read_bytes()).hexdigest()
        if name == "corrupt":
            row["md5"] = "0" * 32            # a publisher checksum the copy does not match
        return dest

    for name, row in picks.items():
        place(name, row)
    # A single-slide TCGA patient in subset B. The split balances slide sizes, so the copy is padded
    # to the real slide's size (trailing bytes after a TIFF are ignored) to keep the split unchanged.
    import split
    in_b = {r["slide_id"] for r in split.select_files(files, "B", split.build_patient_split(files))}
    per_patient = {}
    for r in tcga:
        per_patient.setdefault(r["patient_id"], []).append(r)
    single_b = min((r for r in tcga if r["slide_id"] in in_b and len(per_patient[r["patient_id"]]) == 1),
                   key=lambda r: r["size_bytes"])
    real_size = single_b["size_bytes"]
    dest = place("single_b", single_b)
    with dest.open("ab") as handle:
        handle.truncate(max(real_size, dest.stat().st_size))
    single_b["size_bytes"] = dest.stat().st_size
    single_b["md5"] = hashlib.md5(dest.read_bytes()).hexdigest()
    picks["single_b"] = single_b
    test_manifest = work / "test_manifest.json"
    test_manifest.write_bytes(json.dumps(manifest).encode())
    download.MANIFEST_SHA256 = hashlib.sha256(test_manifest.read_bytes()).hexdigest()
    common_args = ["--root", str(root), "--subset", "full", "--manifest", str(test_manifest),
                   "--log-dir", str(logs)]
    ids = [picks[n]["slide_id"] for n in ("mixed", "faint", "coarse_ok")]

    def config(name, **changes):
        """A copy of configs/tiling.yaml with some values changed."""
        text = (REPO / "configs" / "tiling.yaml").read_text()
        lines = []
        for line in text.splitlines():
            key = line.split(":")[0]
            lines.append(f"{key}: {changes[key]}" if key in changes else line)
        path = work / f"{name}.yaml"
        path.write_text("\n".join(lines) + "\n")
        return path

    def policy(name, **changes):
        """A copy of configs/qc_policy.yaml with some values changed (GrandQC off unless set)."""
        changes = {"grandqc": "false", **changes}
        text = (REPO / "configs" / "qc_policy.yaml").read_text()
        lines = []
        for line in text.splitlines():
            key = line.split(":")[0]
            lines.append(f"{key}: {changes[key]}" if key in changes else line)
        path = work / f"{name}.yaml"
        path.write_text("\n".join(lines) + "\n")
        return path

    # ------------------------------------------------------------------ 1. register
    code = download.main(common_args + ["--register"])
    done = {n: download.slide_progress(root, r)[0] for n, r in picks.items()}
    check("register: exit 1 (rest of the cohort is missing)", code == 1, code)
    check("register: good copies verified", all(done[n] for n in ("mixed", "faint", "coarse_ok", "coarse_bad")), done)
    check("register: copy with the wrong MD5 rejected and left in place",
          not done["corrupt"] and download.paths_for(root, picks["corrupt"])[0].exists())
    receipt = json.loads(download.paths_for(root, picks["coarse_ok"])[3].read_text())
    check("register: CPTAC receipt records no publisher MD5", receipt["publisher_md5_checked"] is False)

    # ------------------------------------------------------------------ 2. tiling
    cfg, sha, _ = common.load_tiling_config(common.DEFAULT_TILING_CONFIG)
    version = cfg["version"]
    code = tile.main(common_args + ["--slides"] + ids + [picks["coarse_bad"]["slide_id"], "--workers", "2"])
    check("tile: exit 1 because the too-coarse slide fails", code == 1, code)
    check("tile: too-coarse slide (2% upsampling) not tiled", not common.tiled(root, picks["coarse_bad"], version, sha))
    check("tile: three slides tiled", all(common.tiled(root, picks[n], version, sha) for n in ("mixed", "faint", "coarse_ok")))

    row = picks["mixed"]
    h5_path, _, mask_path, csv_path, _ = common.output_paths(root, row, version)
    rows = read_csv(csv_path)
    tissue_rows = [r for r in rows if r["role"] == "tissue"]
    with h5py.File(h5_path) as h5:
        attrs = dict(h5.attrs)
        coords = h5["coords"][:]
        first = np.asarray(Image.open(io.BytesIO(h5["tiles"][0].tobytes())))
        first_hash = h5["pixel_sha256"][0].decode()
        n_tiles, grid_index = len(h5["tiles"]), h5["grid_index"][:]
        roles = h5["grid_role"][:]
    check("h5: tiles are 224 x 224 RGB", first.shape == (224, 224, 3), first.shape)
    check("h5: PNG storage is lossless", hashlib.sha256(first.tobytes()).hexdigest() == first_hash)
    check("h5: one stored tile per tissue row, in grid order",
          n_tiles == len(tissue_rows) == attrs["n_tissue_tiles"] == attrs["n_stored_tiles"]
          and [int(r["grid_index"]) for r in tissue_rows] == grid_index.tolist())
    check("h5: every grid position has a role", len(roles) == attrs["n_grid"])
    check("csv: edge tiles scored, not stored",
          any(r["role"] == "edge" for r in rows) and all(r["stored"] == "False" for r in rows if r["role"] == "edge"))
    check("csv: every score column filled", all(r[k] != "" for r in rows for k in slides.SCORE_FIELDS))
    key = "TCGA_" + hashlib.sha256(row["slide_id"].encode()).hexdigest()[:16]
    check("csv: patch IDs follow the gbm_patching scheme", tissue_rows[0]["patch_id"] == f"{key}_{coords[0][0]}_{coords[0][1]}")
    first_scores = slides.tile_scores(first, cfg)
    check("csv: saved scores match a recomputation", all(abs(first_scores[k] - float(tissue_rows[0][k])) < 1e-9
                                                         for k in slides.SCORE_FIELDS))
    check("h5: PNG level 6 recorded", "compress_level 6" in attrs["tile_format"])

    from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
    from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
    mask = (np.asarray(Image.open(mask_path)) > 0).astype(np.uint8)
    wsi = WSIReader.open(download.paths_for(root, row)[0])
    ref = SlidingWindowPatchExtractor(wsi, 224, input_mask=VirtualWSIReader(mask, info=wsi.info, mode="bool"),
                                      resolution=0.5, units="mpp", stride=224, min_mask_ratio=0.5, within_bound=True)
    check("grid: tissue tiles equal TIAToolbox's own mask filter", np.array_equal(ref.coordinate_list[:, :2], coords))
    with h5py.File(common.output_paths(root, picks["faint"], version)[0]) as h5:
        faint = dict(h5.attrs)
    check("mask: floor applied and recorded on the faint slide",
          faint["mask_floor_applied"] and faint["mask_threshold"] == 8 and faint["mask_otsu_threshold"] < 8)
    with h5py.File(common.output_paths(root, picks["coarse_ok"], version)[0]) as h5:
        coarse = dict(h5.attrs)
    check("resolution: 0.8% upsampling recorded", abs(coarse["upsample_pct"] - 0.8) < 1e-6, coarse["upsample_pct"])
    check("records: native OpenSlide version saved", "openslide" in json.loads(coarse["library_versions"]))

    other = os.environ.get("GBM_PATCHING_DIR")
    if other and (Path(other) / "gbm_patching.py").exists():
        sys.path.insert(0, other)
        import gbm_patching as gp
        gwsi = gp.open_slide(download.paths_for(root, row)[0], gp.default_config())
        same = [hashlib.sha256(gwsi.read_rect((int(r["x_target"]), int(r["y_target"])), (224, 224), 0.5, "mpp",
                coord_space="resolution").tobytes()).hexdigest() == r["pixel_sha256"] for r in tissue_rows[::4]]
        gp.close_slide(gwsi)
        check("gbm_patching.py reads identical pixels at matched coordinates", all(same), f"{sum(same)}/{len(same)}")
    else:
        print("SKIP gbm_patching.py comparison (set GBM_PATCHING_DIR to run it)")

    before = h5_path.stat().st_mtime_ns
    code = tile.main(common_args + ["--slides"] + ids + ["--workers", "2"])
    check("rerun: finished slides are not redone", code == 0 and h5_path.stat().st_mtime_ns == before, code)
    code = tile.main(common_args + ["--slides"] + ids + ["--audit", "--check-sources", "--workers", "2"])
    check("audit: clean run over every stored tile, slide files rehashed", code == 0, code)
    audit_report = json.loads(sorted((root / "state" / "tiling" / version).glob("audit_full_*.json"))[-1].read_text())
    check("audit: unstored tiles reread from the slide and hash-checked",
          audit_report["tiles_reread"] == 5 * len(ids) and audit_report["slides_with_errors"] == 0
          and audit_report["sources_checked"], audit_report["tiles_reread"])
    receipt = json.loads(common.output_paths(root, row, version)[4].read_text())
    check("receipt: mask, QC image and source slide hashes recorded",
          all(len(receipt[k]) == 64 for k in ("mask_sha256", "qc_png_sha256", "source_sha256")))

    # ------------------------------------------------------------------ 3. guards and masks
    edited = config("tiling_edited", mask_close_px=3)
    check("guard: config changed without a version bump is refused",
          tile.main(common_args + ["--config", str(edited), "--dry-run"]) == 2)
    check("config: out-of-range value refused",
          tile.main(common_args + ["--config", str(config("bad", png_compress_level=12)), "--dry-run"]) == 2)
    check("slides: a name that matches no slide stops the run",
          tile.main(common_args + ["--slides", ids[0], "not-a-slide", "--dry-run"]) == 2)
    check("config: green hue range checked",
          tile.main(common_args + ["--config", str(config("bad2", green_hue_max=20)), "--dry-run"]) == 2)
    code = tile.main(common_args + ["--slides"] + ids[:2] + ["--masks-only", "--workers", "2"])
    review_csv = read_csv(root / "patches" / version / "mask_review" / "mask_review_full.csv")
    check("masks-only: 3 variants x 2 slides, panels named by file",
          code == 0 and len(review_csv) == 6
          and (root / "patches" / version / "mask_review" / "TCGA" / (Path(row["file_name"]).stem + ".png")).exists())

    # ------------------------------------------------------------------ 4. cap
    for name in ("tiling-cap-a", "tiling-cap-b"):
        tile.main(common_args + ["--slides", ids[0], "--config", str(config(name, version=name, max_tiles_per_slide=20)),
                                 "--workers", "1"])
    cap_a, cap_b = (common.output_paths(root, row, n) for n in ("tiling-cap-a", "tiling-cap-b"))
    cap_rows = read_csv(cap_a[3])
    cap_tissue = [r for r in cap_rows if r["role"] == "tissue"]
    with h5py.File(cap_a[0]) as h5, h5py.File(cap_b[0]) as h5b:
        same_sample = np.array_equal(h5["grid_index"][:], h5b["grid_index"][:])
        n_stored = len(h5["tiles"])
    check("cap: 20 stored, every tissue tile still scored", n_stored == 20
          and sum(r["stored"] == "True" for r in cap_tissue) == 20 and len(cap_tissue) == len(tissue_rows))
    check("cap: same sample on every run", same_sample)
    with loader.TileReader(root, config("tiling-cap-a", version="tiling-cap-a", max_tiles_per_slide=20)) as reader:
        unstored = [r for r in cap_tissue if r["stored"] == "False"][:5]
        ok = [reader.read({**r, "h5": str(cap_a[0].relative_to(root))}).shape == (224, 224, 3) for r in unstored]
    check("loader: tiles beyond the cap reread from the slide with matching hashes", all(ok) and len(ok) == 5)

    # ------------------------------------------------------------------ 5. QC
    check("qc: the default policy uses GrandQC", qc.load_policy()[0]["grandqc"] is True)
    check("qc: unfrozen policy accepted by qc.py", qc.main(["--root", str(root), "--subset", "full", "--log-dir",
                                                            str(logs), "--policy", str(policy("p_default"))]) == 0)
    base = {k: "0" for k in ("near_white_fraction", "near_black_fraction", "dark_fraction", "green_fraction",
                             "glass_fraction")}
    base.update(mean_r="200", mean_g="150", mean_b="200", mean_saturation="80", mean_value="200", gray_median="165",
                laplacian_variance="300")
    default_policy = {**qc.load_policy()[0], "grandqc": False}   # the tile-score rules alone
    unset = {**default_policy, "blur_reject_below": None, "blur_review_below": None}
    check("qc: a flat tile (blur 0) goes to review even with the blur cutoffs unset",
          qc.decide({**base, "laplacian_variance": "0"}, unset) == ("review", "flat_tile"))
    check("qc: normal tile accepted", qc.decide(base, default_policy) == ("accept", "pass"))
    banded = {**default_policy, "blur_reject_below": 50, "blur_review_below": 100}
    check("qc: blur below the reject cutoff rejected; a review band only when blur_review_below is set",
          qc.decide({**base, "laplacian_variance": "70"}, default_policy) == ("reject", "blur")
          and qc.decide({**base, "laplacian_variance": "150"}, default_policy) == ("accept", "pass")
          and qc.decide({**base, "laplacian_variance": "70"}, banded) == ("review", "low_blur"))
    check("qc: green ink above the default cutoff goes to review",
          qc.decide({**base, "green_fraction": "0.05"}, default_policy) == ("review", "green_ink")
          and qc.decide({**base, "green_fraction": "0.01"}, default_policy) == ("accept", "pass"))
    check("qc: malformed, non-finite and out-of-range scores reject",
          all(qc.decide({**base, **bad}, default_policy)[0] == "reject" for bad in (
              {"laplacian_variance": "nan"}, {"laplacian_variance": "inf"}, {"laplacian_variance": "-1"},
              {"mean_value": "256"}, {"glass_fraction": "1.01"}, {"dark_fraction": None},
              {"mean_saturation": ""}, {"mean_g": "bad"}))
          and qc.decide({k: v for k, v in base.items() if k != "mean_b"}, default_policy)
              == ("reject", "invalid_score:mean_b"))
    edge = {**base, "role": "edge"}
    no_edge_accept = {**default_policy, **dict.fromkeys(qc.EDGE_ACCEPT_KEYS)}
    check("qc: an edge tile passes only when clean and within the edge_accept cutoffs",
          qc.decide(edge, default_policy) == ("accept", "edge_accepted")
          and qc.decide({**edge, "glass_fraction": "0.6"}, default_policy) == ("review", "edge")
          and qc.decide({**edge, "dark_fraction": "0.3"}, default_policy) == ("review", "edge")
          and qc.decide({**edge, "green_fraction": "0.05"}, default_policy) == ("review", "edge;green_ink")
          and qc.decide({**edge, "glass_fraction": "0.95"}, default_policy)[0] == "reject"
          and qc.decide(edge, no_edge_accept) == ("review", "edge"))
    check("qc: slide flags for pale slides, slides without tissue tiles and the mask floor",
          qc.slide_flags(default_policy, 20.0, 10, False) == ["pale"]
          and qc.slide_flags(default_policy, 80.0, 0, True) == ["no_tissue", "mask_floor"]
          and qc.slide_flags(default_policy, 80.0, 10, False) == []
          and qc.slide_flags({**default_policy, "pale_slide_saturation_below": None}, 20.0, 10, False) == [])
    strict = {**default_policy, "blur_review_below": 400, "green_fraction_review_above": 0.01,
              "pale_review_below": 100}
    check("qc: optional rules route to review",
          qc.decide({**base, "green_fraction": "0.5"}, strict)[1] == "low_blur;pale;green_ink")
    gq_policy = qc.load_policy()[0]
    clean = {"assessed": 1.0, "tissue": 1.0, "fold": 0.0, "darkspot": 0.0, "pen": 0.0, "bubble": 0.0,
             "oof": 0.0, "background": 0.0, "pen_slide": False, "pale_slide": False}
    check("qc: GrandQC out of focus and pen (on pen slides only) go to review; unassessed tiles skip them",
          qc.decide(base, gq_policy, clean) == ("accept", "pass")
          and qc.decide(base, gq_policy, {**clean, "oof": 0.6}) == ("review", "gq_oof")
          and qc.decide(base, gq_policy, {**clean, "pen": 0.5}) == ("accept", "pass")
          and qc.decide(base, gq_policy, {**clean, "pen": 0.5, "pen_slide": True}) == ("review", "gq_pen")
          and qc.decide(base, gq_policy, {**clean, "oof": 0.9, "assessed": 0.3}) == ("accept", "pass")
          and qc.decide(base, gq_policy) == ("accept", "pass"))
    check("qc: dark spot review (when set) is skipped on pale slides",
          qc.decide(base, {**gq_policy, "darkspot_review_above": 0.1}, {**clean, "darkspot": 0.2})
          == ("review", "gq_darkspot")
          and qc.decide(base, {**gq_policy, "darkspot_review_above": 0.1},
                        {**clean, "darkspot": 0.2, "pale_slide": True}) == ("accept", "pass"))
    check("qc: with GrandQC an edge tile also needs few artifacts and enough assessed area",
          qc.decide(edge, gq_policy, clean) == ("accept", "edge_accepted")
          and qc.decide(edge, gq_policy, {**clean, "fold": 0.3}) == ("review", "edge")
          and qc.decide(edge, gq_policy, {**clean, "assessed": 0.3}) == ("review", "edge;grandqc_unassessed")
          and qc.decide(edge, gq_policy) == ("review", "edge;grandqc_unassessed"))
    check("qc: frozen policy without reviewer refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad", frozen="true"))]) == 2)
    check("qc: negative blur cutoff refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad2", blur_reject_below=-5))]) == 2)
    check("qc: review cutoff below the reject cutoff refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad3", blur_review_below=40))]) == 2)
    check("qc: frozen policy without calibration patients refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad4", frozen="true", reviewer='"tests"', notes='"x"'))]) == 2)
    check("qc: frozen policy with a blur cutoff unset refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad5", frozen="true", reviewer='"tests"', notes='"x"',
                                          calibration_patients=f'["{row["patient_id"]}"]',
                                          blur_reject_below="null"))]) == 2)
    check("qc: edge_accept cutoffs set only in part refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad6", edge_accept_dark_below="null"))]) == 2)
    check("qc: GrandQC without grandqc_min_assessed, or with only one pen cutoff, refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy",
                   str(policy("p_bad7", grandqc="true", grandqc_min_assessed="null"))]) == 2
          and qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy",
                       str(policy("p_bad8", grandqc="true", pen_slide_above="null"))]) == 2)
    edge_counts = read_csv(root / "patches" / version / "qc" / default_policy["version"] / "slides_full.csv")
    check("qc: edge tiles decided (accept, review or reject), counted apart from tissue",
          all(int(s["edge_tiles"]) == int(s["edge_accept"]) + int(s["edge_review"]) + int(s["edge_reject"])
              for s in edge_counts)
          and sum(int(s["edge_tiles"]) for s in edge_counts) > 0)
    faint_summary = next(s for s in edge_counts if s["slide_id"] == picks["faint"]["slide_id"])
    check("qc: the slide summary flags the slide whose mask needed the floor",
          "mask_floor" in faint_summary["slide_flags"].split(";")
          and faint_summary["median_mean_saturation"] != "", faint_summary)

    # ------------------------------------------------------------------ 6. export gates and reviews
    check("export: refuses an unfrozen policy", export.main(common_args[:4] + ["--manifest", str(test_manifest),
                                                                               "--log-dir", str(logs)]) == 2)
    calibration = f'["{row["patient_id"]}"]'
    folds = common.read_folds()
    calibration_fold = int(folds[("TCGA", row["patient_id"])])
    b_fold = int(folds[("TCGA", picks["single_b"]["patient_id"])])
    validation_fold = next(f for f in range(5) if f not in (calibration_fold, b_fold))   # a fold to run
    unknown_calibration = policy("p_unknown", frozen="true", reviewer='"tests"', notes='"x"',
                                 calibration_patients='["TCGA-00-0000"]')
    check("qc: calibration patients must be in the fold map",
          qc.main(common_args[:4] + ["--policy", str(unknown_calibration), "--log-dir", str(logs)]) == 2)
    check("export: calibration patients must be in the fold map", export.main(
        ["--root", str(root), "--subset", "full", "--manifest", str(test_manifest), "--log-dir", str(logs),
         "--policy", str(unknown_calibration)]) == 2)
    check("qc: a CPTAC policy refuses TCGA calibration patients", qc.main(
        common_args[:4] + ["--log-dir", str(logs), "--policy", str(policy(
            "p_cptac_bad", version="qc-cptac-bad", sources="[CPTAC]", frozen="true", reviewer='"tests"',
            notes='"x"', calibration_patients=f'["{row["patient_id"]}"]'))]) == 2)
    frozen = policy("p_frozen", version="qc-test", frozen="true", reviewer='"tests"', notes='"synthetic slides"',
                    calibration_patients=calibration)
    qc_args = ["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy", str(frozen)]
    export_args = ["--root", str(root), "--subset", "full", "--manifest", str(test_manifest),
                   "--log-dir", str(logs), "--policy", str(frozen)]
    check("export: refuses before qc.py ran with this policy", export.main(export_args) == 2)
    check("qc: one frozen policy, calibrated on a TCGA patient from any fold, applied", qc.main(qc_args) == 0)
    check("qc: a frozen policy's folder refuses a different policy",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy",
                   str(policy("p_frozen2", version="qc-test", frozen="true", reviewer='"tests"',
                              notes='"changed"', calibration_patients=calibration))]) == 2)
    third = next(r for r in files if r["patient_id"] == row["patient_id"]
                 and r["slide_id"] not in (row["slide_id"], picks["faint"]["slide_id"]))
    review.sign_off_slide(root, version, "TCGA", third["slide_id"], "exclude", "tests", notes="not available")
    accepted = [d for d in read_csv(qc.decisions_path(root, version, "qc-test", row))
                if d["decision"] == "accept" and d["role"] == "tissue"]
    rejected_by_patch, region_tile = accepted[0], next(r for r in tissue_rows if r["patch_id"] == accepted[1]["patch_id"])
    review.append_patch_review(root, version, rejected_by_patch["patch_id"], "reject", "tests", "fold")
    inside = (int(region_tile["x0"]) + 2, int(region_tile["y0"]) + 2, int(region_tile["x1"]) - 2,
              int(region_tile["y1"]) - 2)   # strictly inside this tile, so neighbours are untouched
    review.append_region(root, version, "TCGA", row["slide_id"], inside, "review", "tests", "pen")
    queue = read_csv(root / "patches" / version / "qc" / "qc-test" / "review_queue_full.csv")
    slide_counts = read_csv(root / "patches" / version / "qc" / "qc-test" / "slides_full.csv")
    check("qc: review queue lists every tissue tile sent to review or rejected",
          len(queue) == sum(int(s["review"]) + int(s["reject"]) for s in slide_counts)
          and all(q["decision"] in ("review", "reject") for q in queue))
    edge_tile = next(d for d in read_csv(qc.decisions_path(root, version, "qc-test", row))
                     if d["role"] == "edge" and d["decision"] == "review")
    review.append_patch_review(root, version, edge_tile["patch_id"], "accept", "tests", notes="interpretable tissue")
    review.sign_off_slide(root, version, "TCGA", row["slide_id"], "approve", "tests", mask=True, focus=True)
    check("review: bad decision refused", _raises(lambda: review.append_patch_review(
        root, version, rejected_by_patch["patch_id"], "maybe", "tests")))
    check("review: exclusion without a reason refused", _raises(lambda: review.sign_off_slide(
        root, version, "TCGA", row["slide_id"], "exclude", "tests")))

    code = export.main(export_args)
    out = sorted((root / "patches" / version / "export" / "qc-test").glob("full_*"))[-1]
    index = {r["patient_id"]: r for r in read_csv(out / "patient_index.csv")}
    model = read_csv(out / "modeling_patches.csv")
    model_ids = {m["patch_id"] for m in model}
    settings = json.loads((out / "export_settings.json").read_text())
    check("export: exit 1 because other patients are blocked", code == 1, code)
    check("export: only fully accounted patients included",
          {p for p, r in index.items() if r["included"] == "True"} == {row["patient_id"], picks["coarse_ok"]["patient_id"]})
    check("export: blocked patients say why",
          "not tiled" in index[picks["coarse_bad"]["patient_id"]]["blocked_reasons"]
          and "not tiled" in index[picks["corrupt"]["patient_id"]]["blocked_reasons"])
    check("export: patch review and region keep tiles out",
          rejected_by_patch["patch_id"] not in model_ids and region_tile["patch_id"] not in model_ids)
    included_decisions = [d for s in ("mixed", "faint", "coarse_ok")
                          for d in read_csv(qc.decisions_path(root, version, "qc-test", picks[s]))]
    qc_accepted = sum(d["decision"] == "accept" for d in included_decisions)
    check("export: every other QC-accepted tile of included patients is in, plus the reinstated edge tile",
          len(model) == qc_accepted - 2 + 1, (len(model), qc_accepted))
    qc_edges = {d["patch_id"] for d in included_decisions if d["role"] == "edge" and d["decision"] == "accept"}
    model_edges = {m["patch_id"] for m in model if m["role"] == "edge"}
    check("export: edge tiles in are those QC accepted plus the one a reviewer accepted, filled after tissue",
          model_edges == qc_edges | {edge_tile["patch_id"]}
          and settings["counts"]["model_edge_tiles"] == len(model_edges)
          and all(m["fill"] == "edge" for m in model if m["role"] == "edge")
          and all(m["fill"] == ("stored" if m["stored"] == "True" else "tissue")
                  for m in model if m["role"] == "tissue"),
          (len(model_edges), len(qc_edges)))
    check("export: rows carry the shared fold",
          {m["fold"] for m in model if m["patient_id"] == row["patient_id"]} == {"0"}
          and {m["fold"] for m in model if m["source"] == "CPTAC"} == {"external"})
    check("export: settings record hashes and the review files",
          settings["modeling_sha256"] == common.sha256_file(out / "modeling_patches.csv")
          and sum(k.startswith("review/") for k in settings["inputs"]) == 3
          and all(not Path(k).is_absolute() for k in settings["inputs"])
          and settings["counts"]["decided_by"].get("qc", 0) > 0)

    # One QC policy per cohort: TCGA and CPTAC apart; CPTAC calibration patients leave the test.
    tcga_policy = policy("p_tcga", version="qc-tcga", sources="[TCGA]", frozen="true", reviewer='"tests"',
                         notes='"x"', calibration_patients=calibration)
    cptac_policy = policy("p_cptac", version="qc-cptac", sources="[CPTAC]", frozen="true", reviewer='"tests"',
                          notes='"x"', calibration_patients=f'["{picks["coarse_ok"]["patient_id"]}"]')
    split_codes = [qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy", str(p)])
                   for p in (tcga_policy, cptac_policy)]
    tcga_slides = read_csv(root / "patches" / version / "qc" / "qc-tcga" / "slides_full.csv")
    cptac_slides = read_csv(root / "patches" / version / "qc" / "qc-cptac" / "slides_full.csv")
    check("qc: a cohort's policy decides only that cohort's slides",
          split_codes == [0, 0] and {s["source"] for s in tcga_slides} == {"TCGA"}
          and {s["source"] for s in cptac_slides} == {"CPTAC"})
    split_args = ["--root", str(root), "--subset", "full", "--manifest", str(test_manifest), "--log-dir", str(logs)]
    check("export: refuses a cohort without a policy, and two policies for one cohort",
          export.main(split_args + ["--policy", str(tcga_policy)]) == 2
          and export.main(split_args + ["--policy", str(tcga_policy), str(frozen)]) == 2)
    export.main(split_args + ["--policy", str(cptac_policy), str(tcga_policy)])
    split_out = sorted((root / "patches" / version / "export" / "qc-tcga+qc-cptac").glob("full_*"))[-1]
    split_index = {r["patient_id"]: r for r in read_csv(split_out / "patient_index.csv")}
    split_settings = json.loads((split_out / "export_settings.json").read_text())
    withheld = split_index[picks["coarse_ok"]["patient_id"]]
    check("export: one policy per cohort; a CPTAC calibration patient is withheld from the test, not blocked",
          [p["version"] for p in split_settings["policies"]] == ["qc-tcga", "qc-cptac"]
          and withheld["included"] == "False" and withheld["qc_calibration"] == "True"
          and withheld["blocked_reasons"] == "" and split_index[row["patient_id"]]["included"] == "True"
          and split_settings["counts"]["patients_withheld_qc_calibration"] == 1
          and all(m["source"] == "TCGA" for m in read_csv(split_out / "modeling_patches.csv"))
          and loader.stale_inputs(split_out) == [], withheld)

    # GrandQC: synthetic masks (all tissue) with painted artifacts; CPTAC has no mask.
    import grandqc
    masks = root / "external" / "grandqc"
    index_rows = {r["slide_id"]: r for r in read_csv(root / "patches" / version / "index_full.csv")}
    v2_rules = {**qc.load_policy()[0], "grandqc": False}
    painted, mask_arrays = {}, {}
    for slide_id, r in index_rows.items():
        if r["source"] != "TCGA":
            continue
        width, height, mpp_x, mpp_y = grandqc.slide_geometry(root / r["h5"])
        mask_arrays[slide_id] = np.ones((round(height * mpp_y), round(width * mpp_x)), dtype=np.uint8)
    def paint(slide_id, tile, label):
        r = index_rows[slide_id]
        width, height, _, _ = grandqc.slide_geometry(root / r["h5"])
        m = mask_arrays[slide_id]
        sx, sy = m.shape[1] / width, m.shape[0] / height
        m[round(int(tile["y0"]) * sy):round(int(tile["y1"]) * sy),
          round(int(tile["x0"]) * sx):round(int(tile["x1"]) * sx)] = label
        painted[tile["patch_id"]] = label
    def tiles_of(pick, role, reason):
        rows = read_csv(root / index_rows[picks[pick]["slide_id"]]["csv"])
        return [t for t in rows if t["role"] == role and qc.decide(t, v2_rules)[1] == reason]
    mixed_ok, faint_ok = tiles_of("mixed", "tissue", "pass"), tiles_of("faint", "tissue", "pass")
    oof_tile, pen_tiles, blank_tile, faint_pen = mixed_ok[0], mixed_ok[1:3], mixed_ok[3], faint_ok[0]
    paint(picks["mixed"]["slide_id"], oof_tile, 6)
    for t in pen_tiles:
        paint(picks["mixed"]["slide_id"], t, 4)          # about 1.5% of the slide: a pen slide
    paint(picks["mixed"]["slide_id"], blank_tile, 0)
    paint(picks["faint"]["slide_id"], faint_pen, 4)      # under 1% of the slide: not a pen slide
    for slide_id, m in mask_arrays.items():
        path = grandqc.mask_path(masks, index_rows[slide_id])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(m, mode="L").save(path)
    gq_code = grandqc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs)])
    gq_status = {r["slide_id"]: r for r in read_csv(root / "patches" / version / "grandqc" / "grandqc_full.csv")}
    gq_rows = {r["patch_id"]: r for sid in mask_arrays
               for r in read_csv(grandqc.output_paths(root, version, index_rows[sid])[0])}
    check("grandqc: per-tile fractions match the painted masks; a slide without a mask is listed, exit 1",
          gq_code == 1 and gq_status[picks["coarse_ok"]["slide_id"]]["status"] == "missing"
          and float(gq_rows[oof_tile["patch_id"]]["oof"]) == 1.0
          and float(gq_rows[blank_tile["patch_id"]]["assessed"]) == 0.0
          and float(gq_rows[mixed_ok[5]["patch_id"]]["tissue"]) == 1.0
          and float(gq_status[picks["mixed"]["slide_id"]]["pen_fraction"]) > 0.01
          > float(gq_status[picks["faint"]["slide_id"]]["pen_fraction"]) > 0, gq_status)
    check("grandqc: a second run skips unchanged slides",
          grandqc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs)]) == 1
          and read_csv(root / "patches" / version / "grandqc" / "grandqc_full.csv")[0]["reason"]
          in ("unchanged", f"no mask at {grandqc.mask_path(masks, index_rows[next(iter(gq_status))])}"))
    gq_frozen = policy("p_gq", version="qc-gq", grandqc="true", frozen="true", reviewer='"tests"', notes='"x"',
                       calibration_patients=calibration)
    gq_qc = qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy", str(gq_frozen)])
    gq_decisions = {d["patch_id"]: d for sid in mask_arrays
                    for d in read_csv(qc.decisions_path(root, version, "qc-gq", index_rows[sid]))}
    check("qc: GrandQC reasons from the masks; the slide without a mask gets no decisions (exit 1)",
          gq_qc == 1 and gq_decisions[oof_tile["patch_id"]]["reasons"] == "gq_oof"
          and all(gq_decisions[t["patch_id"]]["reasons"] == "gq_pen" for t in pen_tiles)
          and gq_decisions[faint_pen["patch_id"]]["decision"] == "accept"
          and gq_decisions[blank_tile["patch_id"]]["decision"] == "accept"
          and not qc.decisions_path(root, version, "qc-gq", index_rows[picks["coarse_ok"]["slide_id"]]).exists(),
          {k: gq_decisions[t["patch_id"]]["reasons"] for k, t in (("oof", oof_tile), ("pen", pen_tiles[0]))})
    gq_receipt = json.loads((root / "patches" / version / "qc" / "qc-gq" / "policy_full.json").read_text())
    gq_summary = {r["slide_id"]: r for r in read_csv(root / "patches" / version / "qc" / "qc-gq" / "slides_full.csv")}
    check("qc: the receipt lists the GrandQC files used and the slide without data",
          len(gq_receipt["grandqc_files"]) == 2 * len(mask_arrays)
          and set(gq_receipt["grandqc_missing"]) == {picks["coarse_ok"]["slide_id"]}
          and int(gq_summary[picks["mixed"]["slide_id"]]["grandqc_unassessed_tissue"]) >= 1)
    export.main(split_args + ["--policy", str(gq_frozen)])
    gq_out = sorted((root / "patches" / version / "export" / "qc-gq").glob("full_*"))[-1]
    gq_index = {r["patient_id"]: r for r in read_csv(gq_out / "patient_index.csv")}
    check("export: a patient whose slide has no GrandQC data is blocked with that reason",
          "no GrandQC data" in gq_index[picks["coarse_ok"]["patient_id"]]["blocked_reasons"]
          and loader.stale_inputs(gq_out) == [], gq_index[picks["coarse_ok"]["patient_id"]]["blocked_reasons"])
    gq_csv = grandqc.output_paths(root, version, index_rows[picks["mixed"]["slide_id"]])[0]
    gq_csv.write_text(gq_csv.read_text() + "\n")
    check("loader: a changed GrandQC file makes the export stale",
          f"{gq_csv.relative_to(root)} changed" in loader.stale_inputs(gq_out))
    bad_mask = grandqc.mask_path(masks, index_rows[picks["faint"]["slide_id"]])
    Image.fromarray(mask_arrays[picks["faint"]["slide_id"]][::2, ::2], mode="L").save(bad_mask)
    grandqc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs)])
    bad_status = {r["slide_id"]: r for r in read_csv(root / "patches" / version / "grandqc" / "grandqc_full.csv")}
    check("grandqc: a mask of the wrong size is refused and its old data dropped",
          bad_status[picks["faint"]["slide_id"]]["status"] == "failed"
          and "mask size" in bad_status[picks["faint"]["slide_id"]]["reason"]
          and grandqc.current(root, version, index_rows[picks["faint"]["slide_id"]])[0] is None,
          bad_status[picks["faint"]["slide_id"]])

    # ------------------------------------------------------------------ 7. loader and Macenko
    check("loader: an incomplete export is refused unless a partial one is intended",
          _raises(lambda: loader.load_modeling(out)))
    tiles = loader.load_modeling(out, allow_partial=True)
    roles = loader.with_roles(tiles, validation_fold)
    held_out = loader.with_roles(tiles, calibration_fold)
    check("loader: one modeling list gives roles for every fold",
          set(roles[roles.patient_id == row["patient_id"]].role) == {"train"}
          and set(held_out[held_out.patient_id == row["patient_id"]].role) == {"validation"}
          and set(roles[roles.source == "CPTAC"].role) == {"external"}
          and _raises(lambda: loader.with_roles(tiles, 5)))
    with loader.TileReader(root) as reader:
        rgb = reader.read(tiles.iloc[0])
        check("loader: stored tile read and hash-checked", rgb.shape == (224, 224, 3))
        bad = tiles.iloc[0].copy()
        bad["pixel_sha256"] = "0" * 64
        check("loader: hash mismatch raises", _raises(lambda: reader.read(bad)))
        stained = {r["patch_id"]: float(r["mean_saturation"]) for r in tissue_rows}
        own = tiles[tiles.patient_id == row["patient_id"]]
        ref_id = max(own.patch_id, key=lambda p: stained.get(p, 0))   # a well-stained tile
        check("macenko: reference from the fold's own validation patient refused",
              _raises(lambda: loader.fit_fold_normalizer(out, calibration_fold, ref_id, reader)))
        normalizer, record = loader.fit_fold_normalizer(out, validation_fold, ref_id, reader)
        normed = loader.normalize_tile(normalizer, reader.read(tiles.iloc[3]), tiles.iloc[3].patch_id,
                                       work / "norm_failures.csv")
        check("macenko: fitted on a training tile and applied",
              normed.shape == (224, 224, 3) and normed.dtype == np.uint8 and record["reference_patch_id"] == ref_id)

        class Broken:
            def transform(self, img):
                raise ValueError("stain matrix failed")
        check("macenko: failure is logged and raised, never passed through",
              _raises(lambda: loader.normalize_tile(Broken(), rgb, "x", work / "norm_failures.csv"))
              and "stain matrix failed" in (work / "norm_failures.csv").read_text())

    check("loader: a current export loads", loader.stale_inputs(out) == [])
    review.append_patch_review(root, version, tiles.iloc[5].patch_id, "reject", "tests", "fold")
    changes = loader.stale_inputs(out)
    check("loader: a review added after the export makes it stale",
          changes == [f"review/{version}/patch_reviews.csv changed"]
          and _raises(lambda: loader.load_modeling(out, allow_partial=True)),
          changes)
    signoff_policy = policy("p_signoff", version="qc-signoff", frozen="true", reviewer='"tests"', notes='"x"',
                            calibration_patients=calibration, require_signoff="true")
    qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy", str(signoff_policy)])
    export.main(["--root", str(root), "--subset", "full", "--manifest", str(test_manifest), "--log-dir", str(logs),
                 "--policy", str(signoff_policy)])
    signed = sorted((root / "patches" / version / "export" / "qc-signoff").glob("full_*"))[-1]
    signed_index = {r["patient_id"]: r for r in read_csv(signed / "patient_index.csv")}
    check("export: with require_signoff, a slide without a sign-off blocks its patient",
          "sign-off" in signed_index[row["patient_id"]]["blocked_reasons"]
          and "sign-off" in signed_index[picks["coarse_ok"]["patient_id"]]["blocked_reasons"])

    # ------------------------------------------------------------------ 8. tampering, export block
    with h5py.File(h5_path, "r+") as h5:
        img = np.asarray(Image.open(io.BytesIO(h5["tiles"][3].tobytes()))).copy()
        img[0, 0] = 255 - img[0, 0]
        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="PNG")
        h5["tiles"][3] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    time.sleep(1.1)
    code = tile.main(common_args + ["--slides"] + ids + ["--audit", "--workers", "2"])
    check("audit: one changed pixel is caught", code == 1, code)
    time.sleep(1.1)
    export.main(export_args)
    out = sorted((root / "patches" / version / "export" / "qc-test").glob("full_*"))[-1]
    index = {r["patient_id"]: r for r in read_csv(out / "patient_index.csv")}
    check("export: a slide that failed the audit blocks its patient",
          index[row["patient_id"]]["included"] == "False" and "audit" in index[row["patient_id"]]["blocked_reasons"])
    qc_png = common.output_paths(root, picks["coarse_ok"], version)[1]
    original = qc_png.read_bytes()
    qc_png.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    time.sleep(1.1)
    tile.main(common_args + ["--slides", picks["coarse_ok"]["slide_id"], "--audit", "--workers", "1"])
    last = json.loads(sorted((root / "state" / "tiling" / version).glob("audit_full_*.json"))[-1].read_text())
    check("audit: a changed QC image is caught", "QC image sha256 differs from receipt" in last["slides"][0]["errors"])
    qc_png.write_bytes(original)

    # ------------------------------------------------------------------ 9. tiles beyond the cap stay out
    cap_cfg = config("tiling-cap-a", version="tiling-cap-a", max_tiles_per_slide=20)
    cap_ids = [ids[0], picks["coarse_ok"]["slide_id"]]
    tile.main(common_args + ["--slides"] + cap_ids + ["--config", str(cap_cfg), "--workers", "2"])
    tile.main(common_args + ["--slides"] + cap_ids + ["--config", str(cap_cfg), "--audit", "--workers", "2"])
    qc.main(qc_args + ["--tiling-config", str(cap_cfg)])
    decided = read_csv(qc.decisions_path(root, "tiling-cap-a", "qc-test", picks["coarse_ok"]))
    dropped = {d["patch_id"] for d in decided
               if d["decision"] == "accept" and d["role"] == "tissue" and d["stored"] == "True"}
    dropped = set(sorted(dropped)[:len(dropped) // 2])   # rejected stored tiles leave room to fill
    for patch_id in dropped:
        review.append_patch_review(root, "tiling-cap-a", patch_id, "reject", "tests", "fold")
    export.main(export_args + ["--tiling-config", str(cap_cfg)])
    cap_out = sorted((root / "patches" / "tiling-cap-a" / "export" / "qc-test").glob("full_*"))[-1]
    cap_model = read_csv(cap_out / "modeling_patches.csv")
    coarse = {r["patient_id"]: r for r in read_csv(cap_out / "patient_index.csv")}[picks["coarse_ok"]["patient_id"]]
    tiers = Counter(export.fill_tier(d) for d in decided if d["decision"] == "accept" and d["patch_id"] not in dropped)
    expected, room = {}, 20
    for tier in export.FILL_ORDER:
        expected[tier] = min(tiers[tier], room)
        room -= expected[tier]
    got = Counter(m["fill"] for m in cap_model if m["slide_id"] == picks["coarse_ok"]["slide_id"])
    check("export: a slide fills the cap with stored tissue, then other tissue, then edge tiles; the rest counted",
          dropped and got["tissue"] > 0 and {k: v for k, v in expected.items() if v} == dict(got)
          and int(coarse["accepted_over_cap"]) == sum(tiers.values()) - sum(got.values()),
          (dict(tiers), dict(got), coarse["accepted_over_cap"]))

    def fake(i, tier):
        return {"slide_id": "s", "patch_id": f"p{i}", "role": "edge" if tier == "edge" else "tissue",
                "stored": str(tier == "stored")}
    pool = [fake(i, t) for i, t in enumerate(["edge"] * 5 + ["tissue"] * 4 + ["stored"] * 3)]
    chosen6, left6 = export.fill_slide(pool, 6)
    chosen10, _ = export.fill_slide(pool, 10)
    check("export: fill_slide takes stored, then tissue, then edge tiles, keeping the original order",
          Counter(t["fill"] for t in chosen6) == {"stored": 3, "tissue": 3} and len(left6) == 6
          and Counter(t["fill"] for t in chosen10) == {"stored": 3, "tissue": 4, "edge": 3}
          and [t["patch_id"] for t in chosen10] == [t["patch_id"] for t in pool
                                                    if t["patch_id"] in {c["patch_id"] for c in chosen10}]
          and len(export.fill_slide(pool, None)[0]) == 12 and export.fill_slide(pool, 6) == (chosen6, left6))
    time.sleep(1.1)
    export.main(export_args + ["--tiling-config", str(cap_cfg)])
    cap_again = read_csv(sorted((root / "patches" / "tiling-cap-a" / "export" / "qc-test").glob("full_*"))[-1]
                         / "modeling_patches.csv")
    check("export: the cap fill is the same on every run",
          [m["patch_id"] for m in cap_again] == [m["patch_id"] for m in cap_model])

    # ------------------------------------------------------------------ 10. packages, combining, sampling
    b_row = picks["single_b"]
    b_args = ["--root", str(root), "--subset", "B", "--manifest", str(test_manifest), "--log-dir", str(logs)]
    a_args = ["--root", str(root), "--subset", "A", "--manifest", str(test_manifest), "--log-dir", str(logs)]
    test_files = json.loads(test_manifest.read_text())["files"]
    in_a = {r["slide_id"] for r in split.select_files(test_files, "A", split.build_patient_split(test_files))}
    a_ids = [s for s in ids if s in in_a]
    b_ids = [s for s in ids if s not in in_a] + [b_row["slide_id"]]
    common.output_paths(root, row, version)[4].unlink()      # re-tile the slide changed in section 8
    for subset_args, subset_ids in ((a_args, a_ids), (b_args, b_ids)):
        tile.main(subset_args + ["--slides"] + subset_ids + ["--workers", "2"])   # also writes index_<subset>
        tile.main(subset_args + ["--slides"] + subset_ids + ["--audit", "--workers", "2"])
    for subset_args in (a_args, b_args):
        qc.main(subset_args[:4] + ["--log-dir", str(logs), "--policy", str(frozen)])
    b_edge = next(d for d in read_csv(qc.decisions_path(root, version, "qc-test", b_row))
                  if d["role"] == "edge" and d["decision"] == "review")
    review.append_patch_review(root, version, b_edge["patch_id"], "accept", "tests", notes="interpretable tissue")
    for subset_args in (a_args, b_args):
        export.main(subset_args + ["--policy", str(frozen)])
    exp_a = sorted((root / "patches" / version / "export" / "qc-test").glob("A_*"))[-1]
    exp_b = sorted((root / "patches" / version / "export" / "qc-test").glob("B_*"))[-1]
    packages = work / "packages"
    pkg_a, pkg_b = packages / "A", packages / "B"
    check("package: an incomplete export is refused without --allow-partial",
          package.main(["--export", str(exp_a), "--out", str(pkg_a), "--log-dir", str(logs)]) == 2 and not pkg_a.exists())
    code_a = package.main(["--export", str(exp_a), "--out", str(pkg_a), "--allow-partial", "--log-dir", str(logs)])
    code_b = package.main(["--export", str(exp_b), "--out", str(pkg_b), "--allow-partial", "--log-dir", str(logs)])
    manifest_b = json.loads((pkg_b / "package_manifest.json").read_text())
    unstored_b = sum(m["stored"] != "True" for m in read_csv(exp_b / "modeling_patches.csv"))
    check("package: built, every model tile not stored at tiling (edge tiles) reread from its slide and stored",
          code_a == code_b == 0 and unstored_b >= 1
          and manifest_b["counts"]["tiles_reread_from_slides"] == unstored_b, (code_a, code_b, unstored_b))
    tiles_a = loader.load_modeling(pkg_a, allow_partial=True)
    with loader.TileReader(pkg_a) as reader:
        ok = [reader.read(r).shape == (224, 224, 3) for _, r in tiles_a.iterrows()]
    check("package: every tile stored and readable with only the package",
          len(tiles_a) == len(read_csv(exp_a / "modeling_patches.csv")) and all(ok)
          and set(tiles_a["stored"]) == {True} and loader.verify_package(pkg_a) == [])
    review.append_patch_review(root, version, b_edge["patch_id"], "reject", "tests", "background")
    check("package: a stale export is refused", package.main(
        ["--export", str(exp_b), "--out", str(packages / "B2"), "--allow-partial", "--log-dir", str(logs)]) == 2)

    combined = packages / "AB"
    combine_args = ["--parts", f"A={pkg_a}", f"B={pkg_b}", "--out", str(combined), "--log-dir", str(logs)]
    check("combine: incomplete parts refused without --allow-partial", combine.main(combine_args) == 2)
    check("combine: a patient in two parts refused", combine.main(
        ["--parts", f"A={pkg_a}", f"A2={pkg_a}", "--out", str(packages / "AA"), "--allow-partial",
         "--log-dir", str(logs)]) == 2)
    check("combine: joined", combine.main(combine_args + ["--allow-partial"]) == 0)
    tiles_ab = loader.load_modeling(combined, allow_partial=True)
    with loader.TileReader(combined) as reader:
        both = [reader.read(tiles_ab[tiles_ab.part == p].iloc[0]).shape == (224, 224, 3) for p in ("A", "B")]
    check("combine: rows of both parts, each read from its own package",
          len(tiles_ab) == len(tiles_a) + manifest_b["counts"]["tiles"] and all(both))
    check("loader: an incomplete combination is refused unless a partial one is intended",
          _raises(lambda: loader.load_modeling(combined)))

    roles = loader.with_roles(tiles_ab, validation_fold)
    bag = loader.training_sample(roles, b_row["patient_id"], epoch=0)
    again = loader.training_sample(roles, b_row["patient_id"], epoch=0)
    other = loader.training_sample(roles, b_row["patient_id"], epoch=1)
    check("sampler: 64 tiles, the same for the same epoch, different for another epoch",
          len(bag) == 64 and list(bag.patch_id) == list(again.patch_id) and list(bag.patch_id) != list(other.patch_id))
    small = loader.training_sample(roles, b_row["patient_id"], epoch=0, count=10)
    check("sampler: no repeats when the patient has enough tiles", small.patch_id.nunique() == 10)
    check("sampler: a validation patient is refused",
          _raises(lambda: loader.training_sample(
              roles.assign(role="validation"), b_row["patient_id"], epoch=0)))
    seen = [pid for chunk in loader.evaluation_batches(tiles_ab, b_row["patient_id"], 16) for pid in chunk.patch_id]
    check("sampler: evaluation visits every tile of the patient once",
          sorted(seen) == sorted(tiles_ab[tiles_ab.patient_id == b_row["patient_id"]].patch_id))

    damaged = sorted((pkg_a / "tiles").rglob("*.h5"))[0]
    data = bytearray(damaged.read_bytes())
    data[len(data) // 2] ^= 1
    damaged.write_bytes(bytes(data))
    check("package: verify_package catches a changed file", any("changed" in p for p in loader.verify_package(pkg_a)))

    # ------------------------------------------------------------------ 11. shell script
    script = REPO / "scripts" / "run_pipeline.sh"
    dry = subprocess.run(["bash", str(script), "--root", str(root), "--subset", "full", "--steps",
                          "register,tile,qc", "--dry-run"], capture_output=True, text=True)
    check("run_pipeline.sh: dry run lists the steps in order", dry.returncode == 0
          and [line.split(":")[0] for line in dry.stdout.splitlines() if line.startswith("==> ") and "done" not in line]
          == ["==> register", "==> tile", "==> qc"], dry.stdout)
    failing = subprocess.run(["bash", str(script), "--root", str(root), "--subset", "full", "--steps", "qc,export",
                              "--python", "false"], capture_output=True, text=True)
    check("run_pipeline.sh: stops at the first failing step", failing.returncode == 1 and "export" not in failing.stdout)
    bogus = subprocess.run(["bash", str(script), "--root", str(root), "--subset", "full", "--steps", "bogus"],
                           capture_output=True, text=True)
    check("run_pipeline.sh: unknown step refused", bogus.returncode == 2)

    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} checks passed")
    if passed == len(results) and not keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"Test data kept in {work}")
    return 0 if passed == len(results) else 1


def _raises(function):
    try:
        function()
    except Exception:
        return True
    return False


if __name__ == "__main__":   # required: tile.py starts worker processes with spawn
    sys.exit(main())
