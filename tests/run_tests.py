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
    import make_slides
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
    for name, row in picks.items():
        dest = root / "raw" / row["source"] / row["patient_id"] / row["file_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(made["mixed" if name == "corrupt" else name], dest)
        row["size_bytes"] = dest.stat().st_size
        if row["source"] == "TCGA":
            row["md5"] = hashlib.md5(dest.read_bytes()).hexdigest()
        if name == "corrupt":
            row["md5"] = "0" * 32            # a publisher checksum the copy does not match
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
    code = tile.main(common_args + ["--slides"] + ids + ["--audit", "--workers", "2"])
    check("audit: clean run over every stored tile", code == 0, code)

    # ------------------------------------------------------------------ 3. guards and masks
    edited = config("tiling_edited", mask_close_px=3)
    check("guard: config changed without a version bump is refused",
          tile.main(common_args + ["--config", str(edited), "--dry-run"]) == 2)
    check("config: out-of-range value refused",
          tile.main(common_args + ["--config", str(config("bad", png_compress_level=12)), "--dry-run"]) == 2)
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
    check("qc: unfrozen policy accepted by qc.py", qc.main(["--root", str(root), "--subset", "full",
                                                            "--log-dir", str(logs)]) == 0)
    base = {k: "0" for k in ("near_white_fraction", "near_black_fraction", "dark_fraction", "green_fraction",
                             "glass_fraction")}
    base.update(mean_r="200", mean_g="150", mean_b="200", mean_saturation="80", mean_value="200",
                laplacian_variance="300")
    default_policy = qc.load_policy()[0]
    check("qc: a flat tile (blur 0) goes to review even with min_blur unset",
          qc.decide({**base, "laplacian_variance": "0"}, default_policy) == ("review", "flat_tile"))
    check("qc: normal tile accepted", qc.decide(base, default_policy) == ("accept", "pass"))
    strict = {**default_policy, "min_blur": 400, "green_fraction_review_above": 0.01, "pale_review_below": 100}
    check("qc: optional rules route to review",
          qc.decide({**base, "green_fraction": "0.5"}, strict)[1] == "low_blur;pale;green_ink")
    check("qc: frozen policy without reviewer refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad", frozen="true"))]) == 2)
    check("qc: negative blur cutoff refused",
          qc.main(["--root", str(root), "--subset", "full", "--log-dir", str(logs),
                   "--policy", str(policy("p_bad2", min_blur=-5))]) == 2)

    # ------------------------------------------------------------------ 6. export gates and reviews
    check("export: refuses an unfrozen policy", export.main(common_args[:4] + ["--manifest", str(test_manifest),
                                                                               "--log-dir", str(logs)]) == 2)
    frozen = policy("p_frozen", version="qc-test", frozen="true", reviewer='"tests"', notes='"synthetic slides"')
    qc_args = ["--root", str(root), "--subset", "full", "--log-dir", str(logs), "--policy", str(frozen)]
    export_args = ["--root", str(root), "--subset", "full", "--manifest", str(test_manifest),
                   "--log-dir", str(logs), "--policy", str(frozen)]
    check("export: refuses before qc.py ran with this policy", export.main(export_args) == 2)
    qc.main(qc_args)
    third = next(r for r in files if r["patient_id"] == row["patient_id"]
                 and r["slide_id"] not in (row["slide_id"], picks["faint"]["slide_id"]))
    review.sign_off_slide(root, version, "TCGA", third["slide_id"], "exclude", "tests", notes="not available")
    accepted = [d for d in read_csv(qc.decisions_path(root, version, "qc-test", row)) if d["decision"] == "accept"]
    rejected_by_patch, region_tile = accepted[0], next(r for r in tissue_rows if r["patch_id"] == accepted[1]["patch_id"])
    review.append_patch_review(root, version, rejected_by_patch["patch_id"], "reject", "tests", "fold")
    inside = (int(region_tile["x0"]) + 2, int(region_tile["y0"]) + 2, int(region_tile["x1"]) - 2,
              int(region_tile["y1"]) - 2)   # strictly inside this tile, so neighbours are untouched
    review.append_region(root, version, "TCGA", row["slide_id"], inside, "review", "tests", "pen")
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
    qc_accepted = sum(d["decision"] == "accept" for s in ("mixed", "faint", "coarse_ok")
                      for d in read_csv(qc.decisions_path(root, version, "qc-test", picks[s])))
    check("export: every other QC-accepted tile of included patients is in", len(model) == qc_accepted - 2,
          (len(model), qc_accepted))
    check("export: rows carry the shared fold",
          {m["fold"] for m in model if m["patient_id"] == row["patient_id"]} == {"0"}
          and {m["fold"] for m in model if m["source"] == "CPTAC"} == {"external"})
    check("export: settings record hashes and the review files",
          settings["modeling_sha256"] == common.sha256_file(out / "modeling_patches.csv")
          and len(settings["review_files"]) == 3 and settings["counts"]["decided_by"].get("qc", 0) > 0)

    # ------------------------------------------------------------------ 7. loader and Macenko
    tiles = loader.load_modeling(out)
    roles0, roles1 = loader.with_roles(tiles, 0), loader.with_roles(tiles, 1)
    check("loader: roles per fold", set(roles0[roles0.patient_id == row["patient_id"]].role) == {"validation"}
          and set(roles1[roles1.patient_id == row["patient_id"]].role) == {"train"}
          and set(roles1[roles1.source == "CPTAC"].role) == {"external"})
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
              _raises(lambda: loader.fit_fold_normalizer(out, 0, ref_id, reader)))
        normalizer, record = loader.fit_fold_normalizer(out, 1, ref_id, reader)
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

    # ------------------------------------------------------------------ 9. shell script
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
