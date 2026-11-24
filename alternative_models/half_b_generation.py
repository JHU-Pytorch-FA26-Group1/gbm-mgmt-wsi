"""Reuse recorded Half A inference engines for matched Half B generation.

Only source preparation/portable-package validation and orchestration are new.
Inference functions are imported from byte-checked, unchanged Half A snapshots.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from . import half_b_package as pre

NAMES = ("his_mmdm_he_augment", "his_mmdm_batch", "his_mmdm_corners", "his_mmdm_resize_matched")
DEFAULT_BASE = Path("/workspace/persist/crivas4-microscopy")
PROTOCOL = "half-b-matched-half-a-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return pre.sha256(Path(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def locations(base):
    pilot = base / "GBM_MGMT/hismmdm-pilot"
    return dict(pilot=pilot, root=pilot / "half-B-fold-0-matched-32000",
                a_original=pilot / "fold-0-balanced-4000", a_crop=pilot / "fold-0-corners-16000",
                a_extension=pilot / "fold-0-resize-matched-additional12000")


def find_code(base, relative, expected):
    """Prefer recorded immutable job code; current checkout is a checked fallback."""
    pilot = locations(base)["pilot"]
    name = Path(relative).name
    candidates = sorted(pilot.glob(f"*/runai/*/code/{name}"))
    candidates += [base / "gbm-mgmt-wsi" / relative, base / "gbm-mgmt-wsi-alternative-models" / relative]
    for path in candidates:
        if path.is_file() and digest(path) == expected:
            return path
    raise ValueError(f"No Half A code snapshot matches its recorded hash: {name}")


def capture_a(base, repo, target):
    paths = locations(base)
    original = read(paths["a_original"] / "batch_plan.json")
    extension = read(paths["a_extension"] / "batch_plan.json")
    crop_run = read(paths["a_crop"] / "generated/shard-0/batch_run.json")
    resize_run = read(paths["a_original"] / "generated/shard-0/batch_run.json")
    require(original["source_count"] == 2000 and original["sources_per_class"] == 1000
            and original["variants_per_source"] == 2 and original["synthetic_count"] == 4000
            and original["held_out_fold"] == 0 and original["eta"] == .3,
            "Half A original dimensions/settings differ from the agreed design")
    sig = crop_run["signature"]
    core_hash = extension["core_sha256"]
    require(core_hash == resize_run["signature"]["wrapper_sha256"],
            "Half A original/extension resize wrappers differ; inspect before cloning")
    require(extension["legacy_sha256"] == resize_run["signature"]["batch_script_sha256"]
            and digest(paths["a_original"] / "batch_plan.json") == resize_run["signature"]["batch_plan_sha256"]
            and digest(paths["a_crop"] / "batch_plan.json") == sig["batch_plan_sha256"]
            and sig["config"] == resize_run["signature"]["model_flags"], "Half A engine/input records disagree")
    require(sig["batch_size"] in (4, 8, 16), "Unexpected Half A production crop batch size")
    hashes = dict(zip(NAMES, (core_hash, extension["legacy_sha256"],
                              sig["runner_sha256"], extension["preparer_sha256"])))
    target.mkdir(parents=True)
    for name, expected in hashes.items():
        source = find_code(base, f"experiments/hismmdm-pilot/{name}.py", expected)
        shutil.copyfile(source, target / f"{name}.py")
    # Crop A may have a different wrapper revision (for example CLI/review
    # validation changes). Keep its actual recorded file, not a guessed common one.
    crop_core = find_code(base, "experiments/hismmdm-pilot/his_mmdm_he_augment.py", sig["wrapper_sha256"])
    shutil.copyfile(crop_core, target / "crop_core.py")
    receipts = []
    for shard in (0, 1):
        name = f"shard-{shard}-light-qc-v4-receipt.json"
        candidate = base / "gbm-mgmt-wsi/experiments/hismmdm-pilot" / name
        if not candidate.is_file():
            candidate = repo / "experiments/hismmdm-pilot" / name
        r = read(candidate)
        require(r["complete"] is True and r["protocol"] == "synthetic-light-v4" and r["shard"] == shard
                and r["images"] == sum(r["counts"].values()) == 2000
                and r["effective_policy"]["blur_reject_below"] == 10, "Expected Half A's final blur-10 receipts")
        receipts.append(r)
        write(target / name, r)
    require(all(receipts[0][k] == receipts[1][k] for k in
                ("parent_policy_sha256", "tiling_sha256", "effective_policy", "code_sha256", "runtime")),
            "Half A shards used different QC inputs or runtimes")
    qc = receipts[0]
    qc_name = next(k for k in qc["code_sha256"] if k.endswith("/qc_synthetics_light.py"))
    qc_source = find_code(base, "experiments/hismmdm-pilot/qc_synthetics_light.py", qc["code_sha256"][qc_name])
    shutil.copyfile(qc_source, target / "qc_synthetics_light.py")
    # This evidence is available while A's later shards are still running.
    return dict(seed=original["seed"], verify_per_class=original["verify_per_class"], fold=0,
                crop_batch_size=sig["batch_size"], engine_sha256=hashes,
                crop_core_sha256=sig["wrapper_sha256"],
                a_resize_runtime=resize_run["signature"]["runtime"], a_crop_runtime=sig["runtime"],
                a_model_flags=resize_run["signature"]["model_flags"],
                qc_receipt_sha256=digest(target / "shard-0-light-qc-v4-receipt.json"))


@contextmanager
def engines(code, portable=False):
    """Isolated imports; the explicit adapter changes only source validation.

    No source file or global original module is modified. Dependency slots of
    these newly imported engine instances point at a proxy; the inference loops
    themselves remain the exact functions from Half A's recorded files.
    """
    saved = {name: sys.modules.pop(name, None) for name in NAMES}
    sys.path.insert(0, str(code))
    try:
        core, legacy, crop, matched = [importlib.import_module(name) for name in NAMES]
        if (code / "crop_core.py").exists():
            spec = importlib.util.spec_from_file_location("_half_b_recorded_crop_core", code / "crop_core.py")
            crop_core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(crop_core)
            crop.core = crop_core
        else:
            crop_core = core
        if portable:
            class PortableCore:
                def __getattr__(self, name):
                    return getattr(core, name)

                def validated_inputs(self, args):
                    return portable_inputs(args, core)
            proxy = PortableCore()
            legacy.core = matched.core = proxy
            crop.core = crop_core
            matched.legacy = legacy
        yield SimpleNamespace(core=core, legacy=legacy, crop=crop, matched=matched)
    finally:
        sys.path.remove(str(code))
        for name in NAMES:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


def portable_inputs(args, core):
    """Equivalent lineage/weight/pixel gates using real package QC provenance.

    Tissue fraction from a whole-slide mask is unavailable in portable packages.
    It is deliberately neither invented nor recomputed with a different method.
    """
    from PIL import Image
    refs = args.refs
    record = read(refs / "references.json")
    require(record.get("source_qc_evidence") == "complete_verified_qc_v5_package"
            and record["held_out_fold"] == args.fold and args.phase == "batch", "Wrong portable source protocol")
    for name, key in (("input_manifest.csv", "manifest_sha256"),
                      ("package_source_rows.csv", "source_rows_sha256"),
                      ("package_manifest.json", "package_manifest_sha256")):
        core.require_hash(refs / name, record[key])
    for name, key in (("patient_folds.csv", "folds"), ("patient_labels.csv", "labels")):
        core.require_hash(refs / name, record["inputs_sha256"][key])
    package = read(refs / "package_manifest.json")
    require(package["kind"] == "package" and package["subset"] == "B" and package["complete"] is True,
            "Wrong/incomplete package provenance")
    allowed = core.training_ids(refs / "patient_folds.csv", args.fold)
    assigned = {r["patient_id"]: r["fold"] for r in core.read_csv(refs / "patient_folds.csv", ("patient_id", "fold"))}
    rows = core.read_csv(refs / "input_manifest.csv", ("reference_id", "patient_id", "patch_id", "image_path"))
    source = core.unique_by(core.read_csv(refs / "package_source_rows.csv", ("patch_id",)), "patch_id")
    labels = {r["patient_id"]: core.label_status(r) for r in core.read_csv(refs / "patient_labels.csv", ("patient_id",))}
    core.unique_by(rows, "reference_id")
    core.unique_by(rows, "patch_id")
    require(len(rows) == len(source) <= 2000, "Invalid portable reference count")
    for row in rows:
        r = source[row["patch_id"]]
        require(row["source"] == r["source"] == "TCGA" and row["patient_id"] == r["patient_id"]
                and row["patient_id"] in allowed and row["assigned_fold"] == r["fold"] == assigned[row["patient_id"]]
                and r["role"] == "tissue" and r["stored"] == "True"
                and r["h5"] in package["files"] and row["slide_id"] == r["slide_id"]
                and row["pixel_sha256"] == r["pixel_sha256"]
                and row["mgmt_patient_label"] == labels[row["patient_id"]], "Invalid portable training lineage")
        require(row["reference_id"].startswith("ref")
                and row["reference_id"][3:].isdigit() and len(row["reference_id"]) == 9
                and row["image_path"] == f"originals/{row['reference_id']}.png"
                and float(row["source_mpp"]) == .5, "Invalid reference path/geometry")
        path = refs / row["image_path"]
        core.require_hash(path, row["source_sha256"])
        with Image.open(path) as im:
            require(im.format == "PNG" and im.mode == "RGB" and im.size == (224, 224)
                    and hashlib.sha256(im.tobytes()).hexdigest() == row["pixel_sha256"], "Original pixel mismatch")
    weights = read(args.weights / "weights.json")
    require(weights.get("family") == "HMU-1st" and weights.get("archive_md5") == core.ARCHIVE_MD5
            and weights.get("filename") == core.WEIGHT_NAME and weights.get("sha256") == core.WEIGHT_SHA256
            and tuple(Path(weights.get("archive_member", "")).parts[-3:]) == core.WEIGHT_MEMBER,
            "Wrong checkpoint receipt")
    checkpoint = args.weights / core.WEIGHT_NAME
    core.require_hash(checkpoint, core.WEIGHT_SHA256)
    return checkpoint, weights, rows, record, sorted(allowed)


def prepare_sources(a, engine, original_plan, stage, per_class=1000):
    """Same Half A patient-quota, slide/block draw and repeat-check selection."""
    import random
    from PIL import Image
    sys.path.insert(0, str(a.repo / "data_prep"))
    try:
        import common
        import loader
        import numpy as np
        import pandas as pd
        package = a.base / "packages/B_qc-v5"
        manifest = read(package / "package_manifest.json")
        folds_path = a.repo / "manifests/patient_folds.csv"
        config = a.repo / "configs/tiling.yaml"
        cfg, cfg_hash, _ = common.load_tiling_config(config)
        require((cfg["tile_px"], cfg["target_mpp"]) == (224, .5)
                and manifest["subset"] == "B" and manifest["complete"] is True
                and manifest["folds_sha256"] == digest(folds_path) and manifest["config_sha256"] == cfg_hash,
                "Half B package geometry/folds differ")
        labels_path = locations(a.base)["a_original"] / "references/patient_labels.csv"
        labels = {r["patient_id"]: engine.core.label_status(r)
                  for r in engine.core.read_csv(labels_path, ("patient_id",))}
        table = loader.load_modeling(package)
        require(not table.patch_id.duplicated().any(), "Duplicate packaged patch IDs")
        source_table = table[table.source.eq("TCGA") & table.role.eq("tissue")].copy()
        folded = loader.with_roles(source_table, 0)
        folded = folded[folded.role.eq("train")]
        folds = common.read_folds(folds_path)
        require(all(folds.get((r.source, r.patient_id)) == r.fold for r in folded.itertuples()), "Modeling fold mismatch")
        canonical = {r["patient_id"]: r["mgmt_status"].lower() for r in pre.read_rows(a.repo / "manifests/patient_labels.csv")}
        require(all(p in labels and labels[p] == canonical[p] for p in folded.patient_id.unique()), "Molecular label mismatch")
        chosen, allocation = engine.legacy.choose_sources(folded, labels, per_class, original_plan["seed"], 0, loader)
        repeat_ids = set()
        for label in engine.legacy.LABELS:
            group = [r for r in chosen if r["mgmt_patient_label"] == label]
            patients = sorted({r["patient_id"] for r in group})
            random.Random(engine.legacy.seed_for(original_plan["seed"], label, "repeat")).shuffle(patients)
            queues = {p: [r for r in group if r["patient_id"] == p] for p in patients}
            wanted = min(original_plan["verify_per_class"], len(group))
            selected = 0
            while selected < wanted:
                for patient in patients:
                    if queues[patient]:
                        repeat_ids.add(queues[patient].pop(0)["patch_id"])
                        selected += 1
                        if selected == wanted:
                            break
        refs = stage / "references"
        (refs / "originals").mkdir(parents=True)
        for src, name in ((folds_path, "patient_folds.csv"), (labels_path, "patient_labels.csv"),
                          (package / "package_manifest.json", "package_manifest.json")):
            shutil.copyfile(src, refs / name)
        original_rows = source_table.set_index("patch_id", drop=False)
        sources, rows = [], []
        with loader.TileReader(package, tiling_config=config) as reader:
            for i, r in enumerate(chosen, 1):
                ref = f"ref{i:06d}"
                source = original_rows.loc[r["patch_id"]].to_dict()
                sources.append(source)
                rgb = reader.read(source)
                require(rgb.shape == (224, 224, 3) and rgb.dtype.name == "uint8", "Wrong source RGB geometry")
                require(hashlib.sha256(rgb.tobytes()).hexdigest() == source["pixel_sha256"], "Package source pixels differ")
                image_path = f"originals/{ref}.png"
                Image.fromarray(rgb).save(refs / image_path)
                rows.append(dict(reference_id=ref, source="TCGA", patient_id=r["patient_id"], assigned_fold=r["fold"],
                    slide_id=r["slide_id"], patch_id=r["patch_id"], image_path=image_path,
                    source_sha256=digest(refs / image_path), pixel_sha256=r["pixel_sha256"],
                    mgmt_patient_label=r["mgmt_patient_label"], source_mpp=.5, source_review_status="pending",
                    workload_shard=r["workload_shard"], source_x_target=r["x_target"], source_y_target=r["y_target"],
                    repeat_checks=r["patch_id"] in repeat_ids, morphology="", qc_defect_flags="",
                    source_qc_evidence="complete_verified_qc_v5_package", source_subset="B",
                    qc_decided_by=source["decided_by"]))
        core = engine.core
        core.save_csv(refs / "input_manifest.csv", rows)
        core.save_csv(refs / "package_source_rows.csv", sources)
        core.save_csv(stage / "patient_allocation.csv", allocation)
        prep = dict(held_out_fold=0, batch_mode=True, preview_only=True,
                    manifest_sha256=digest(refs / "input_manifest.csv"),
                    source_rows_sha256=digest(refs / "package_source_rows.csv"),
                    package_manifest_sha256=digest(refs / "package_manifest.json"),
                    source_qc_evidence="complete_verified_qc_v5_package",
                    unavailable_source_scores="Whole-slide mask fraction and original score CSVs absent; no values invented",
                    inputs_sha256=dict(folds=digest(folds_path), labels=digest(labels_path), tiling=digest(config),
                                       modeling=digest(package / "modeling_patches.csv")),
                    loader_sha256=digest(Path(loader.__file__)),
                    common_sha256=digest(Path(common.__file__)),
                    repo_commit=subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip(),
                    preparation_runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, pillow=Image.__version__),
                    modeling_folder=str(package), prepared_at=now())
        core.save_json(refs / "references.json", prep)
        plan = dict(protocol="frozen-hmu1st-he-resize-v1", held_out_fold=0, seed=original_plan["seed"], eta=.3,
                    source_count=len(rows), variants_per_source=2, synthetic_count=2*len(rows), sources_per_class=per_class,
                    shard_count=2, verify_per_class=original_plan["verify_per_class"], source_subset="B",
                    source_budget="TCGA training tissue only; same Half A patient/slide/block sampler",
                    references_sha256=digest(refs / "references.json"),
                    allocation_sha256=digest(stage / "patient_allocation.csv"), source_score_hashes={},
                    adapter_sha256=digest(__file__), source_review_status="pending", review_status="pending",
                    training_eligible=False, created_at=now())
        core.save_json(stage / "batch_plan.json", plan)
        engine.legacy.validate_batch(stage)
        return rows
    finally:
        sys.path.remove(str(a.repo / "data_prep"))


def prepare(a):
    from .shared_half_b import LocalPackage, validate_manifest
    paths = locations(a.base)
    root = paths["root"]
    require(not root.exists() and not root.is_symlink(), "Half B plan exists; use status/validate, never overwrite")
    package = a.base / "packages/B_qc-v5"
    download = a.base / "experiments/alternative-models/half-b-onedrive/receipt.json"
    receipt = read(download)
    manifest_path = package / "package_manifest.json"
    require(receipt["complete"] is True and receipt["manifest_sha256"] == digest(manifest_path),
            "Expected the completed, hash-verified direct download receipt")
    manifest = validate_manifest(manifest_path.read_bytes(), a.base / "packages/A_qc-v5", a.repo)
    population = pre.population(LocalPackage(package), "", manifest, a.repo)
    require(population["enough_for_1000_distinct_sources_per_class"], "Insufficient eligible source tiles")
    paths["pilot"].mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="half-b.prepare-", dir=paths["pilot"]) as temporary:
        stage = Path(temporary) / "batch"
        stage.mkdir()
        protocol = capture_a(a.base, a.repo, stage / "templates")
        original_plan = read(paths["a_original"] / "batch_plan.json")
        original = stage / "resize-original-4000"
        original.mkdir()
        with engines(stage / "templates", portable=True) as e:
            e.legacy.validate_batch(paths["a_original"])
            e.crop.validated(paths["a_crop"])
            rows = prepare_sources(a, e, original_plan, original)
            a_rows = e.core.read_csv(paths["a_original"] / "references/input_manifest.csv", ("patient_id", "patch_id"))
            require(not ({r["patch_id"] for r in rows} & {r["patch_id"] for r in a_rows})
                    and not ({r["patient_id"] for r in rows} & {r["patient_id"] for r in a_rows}),
                    "Half A and Half B source identities overlap")
            # Published absolute lineage paths must never refer to a temporary staging name.
            # Crop preparation is performed after publication; its own writer is atomic.
        protocol.update(protocol=PROTOCOL, base=str(a.base), repo=str(a.repo), source_subset="B",
                        source_count=len(rows), planned_resize=16000, planned_crop=16000,
                        original_plan_sha256=digest(original / "batch_plan.json"),
                        adapter_sha256=digest(__file__), download_receipt_sha256=digest(download),
                        population=population, created_at=now(), source_review_status="pending",
                        review_status="pending", training_eligible=False)
        write(stage / "protocol.json", protocol)
        stage.rename(root)
    # Both arms derive from one source selection, not two independent draws.
    finish_crop_preparation(a.base)
    validate(a.base)
    print(f"Prepared Half B: 2,000 originals -> 16,000 resize + 16,000 crop variants\n{root}")


def finish_crop_preparation(base):
    root = locations(base)["root"]
    if not (root / "corners-16000").exists():
        with engines(root / "templates", portable=True) as e:
            e.crop.prepare(SimpleNamespace(parent=root / "resize-original-4000", out=root / "corners-16000"))


def validate(base):
    root = locations(base)["root"]
    p = read(root / "protocol.json")
    require(p["protocol"] == PROTOCOL and p["source_count"] == 2000
            and p["planned_resize"] == p["planned_crop"] == 16000
            and p["adapter_sha256"] == digest(__file__), "Wrong/changed Half B protocol/adapter")
    for name, expected in p["engine_sha256"].items():
        require(digest(root / "templates" / f"{name}.py") == expected, f"Recorded engine changed: {name}")
    require(digest(root / "templates/crop_core.py") == p["crop_core_sha256"], "Recorded crop core changed")
    require(digest(root / "templates/shard-0-light-qc-v4-receipt.json") == p["qc_receipt_sha256"], "Half A QC evidence changed")
    require(digest(root / "resize-original-4000/batch_plan.json") == p["original_plan_sha256"], "Source plan changed")
    with engines(root / "templates", portable=True) as e:
        plan, rows = e.legacy.validate_batch(root / "resize-original-4000")
        require(plan["seed"] == p["seed"] and plan["verify_per_class"] == p["verify_per_class"], "Changed selection settings")
        e.crop.validated(root / "corners-16000")
    return p


def runtime_matches(expected, crop=False):
    import numpy as np
    from PIL import Image
    import torch
    require(torch.cuda.is_available(), "CUDA is unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    observed = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                    numpy=np.__version__, pillow=Image.__version__, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name(0), cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    observed["capability" if crop else "compute_capability"] = list(torch.cuda.get_device_capability(0))
    if crop:
        observed["tf32"] = False
    require(observed == expected, f"Half B runtime differs from Half A; inspect before generation: {observed}")


def a_complete(base, arm):
    """Read-only completion gate; never rewrites Half A's combined manifests."""
    paths = locations(base)
    if arm == "crop":
        for shard in (0, 1):
            folder = paths["a_crop"] / f"generated/shard-{shard}"
            require((folder / "completed.json").is_file(), f"Half A crop shard {shard} is not finalized; no Half B job submitted")
            done = read(folder / "completed.json")
            require(done["phase"] == "generation" and done["outputs"] == 8000, "Half A crop is not complete")
            require(digest(folder / "generation_manifest.csv") == done["manifest_sha256"], "Half A crop manifest changed")
    else:
        require((paths["a_extension"] / "completed.json").is_file(),
                "Half A matched resize is not finalized; no Half B job submitted")
        done = read(paths["a_extension"] / "completed.json")
        require(done["matched_total"] == 16000
                and digest(paths["a_extension"] / "matched_resize_16000.csv") == done["manifest_sha256"],
                "Half A matched resize is not complete")


def scheduler_gate(text, arm):
    """Parse the installed CLI's already-observed default table, fail closed.

    Only our hismmdm jobs count here; never operate on other users' workloads.
    Requested allocations of Pending jobs count too, avoiding queued duplicates.
    """
    import re
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    require("Workload" in text and "GPU" in text, "Unrecognized Run:ai workload table")
    active = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("hismmdm-"):
            continue
        require(len(fields) >= 7 and fields[1] == "Training", "Unrecognized hismmdm workload row")
        name, state = fields[0], fields[3]
        try:
            allocated = float(fields[-1])
        except ValueError:
            raise ValueError("Unrecognized GPU allocation in workload table") from None
        if state in {"Completed", "Failed", "Stopped"} and allocated == 0:
            continue
        require(not name.startswith(f"hismmdm-halfb-{arm}-"), "A Half B job for this design is already active/queued")
        if "corners" in name or "halfb-crop" in name:
            allocation, kind = 1., "crop"
        elif "resize-matched" in name or "halfb-resize" in name:
            allocation, kind = .5, "resize"
        else:
            raise ValueError(f"Inspect this active generation workload before launching: {name}")
        require(kind != arm, f"Half A {arm} workload has not released its allocation: {name} ({state})")
        active.append((name, allocation))
    requested = 1. if arm == "crop" else .5
    require(sum(v for _, v in active) + requested <= 1.5, "Would exceed the agreed 1.5-GPU generation allocation")
    return dict(arm=arm, requested_gpu=requested, existing_generation_jobs=active)


def a_status(base):
    paths = locations(base)
    for title, folders, original in (
        ("Half A crop", [(paths["a_crop"] / f"generated/shard-{s}", 8000) for s in (0, 1)], 0),
        ("Half A matched resize", [(paths["a_extension"] / f"generated/shard-{s}", 6000) for s in (0, 1)], 4000)):
        total = original
        for folder, budget in folders:
            done, progress = folder / "completed.json", folder / "progress.json"
            if done.is_file():
                count = read(done)["outputs"]
            elif progress.is_file():
                r = read(progress)
                count = r.get("completed_variants", r.get("completed_outputs", 0))
                print(f"{title} / {folder.name}: updated {r.get('updated_at')}")
            else:
                count = 0
            total += count
            print(f"{title} / {folder.name}: {count}/{budget}; finalized={done.is_file()}")
        print(f"{title}: {total}/16000")


def worker(a):
    root = locations(a.base)["root"]
    p = validate(a.base)
    a_complete(a.base, a.arm)
    runtime_matches(p["a_crop_runtime"] if a.arm == "crop" else p["a_resize_runtime"], crop=a.arm == "crop")
    weights = locations(a.base)["pilot"] / "artifacts/hmu-1st"
    with (root / f".{a.arm}.worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with engines(root / "templates", portable=True) as e:
            require(e.core.release_settings(a.base / "His-MMDM") == p["a_model_flags"], "Half A/B model flags differ")
            if a.arm == "crop":
                batch = root / "corners-16000"
                for shard in (0, 1):
                    e.crop.worker(SimpleNamespace(batch=batch, upstream=a.base / "His-MMDM", weights=weights,
                        shard=shard, batch_size=p["crop_batch_size"], limit_groups=0, benchmark_out=None))
                e.crop.inspect(SimpleNamespace(batch=batch, command="verify"))
            else:
                parent, batch = root / "resize-original-4000", root / "resize-additional12000"
                for shard in (0, 1):
                    e.legacy.worker(SimpleNamespace(batch=parent, upstream=a.base / "His-MMDM", weights=weights,
                                                    shard=shard, device="cuda:0"))
                # Same original-01/02 + extension-03..08 execution and replay checks as A.
                if not batch.exists():
                    e.matched.prepare(SimpleNamespace(parent=parent, crop=root / "corners-16000", out=batch))
                for shard in (0, 1):
                    e.matched.worker(SimpleNamespace(batch=batch, upstream=a.base / "His-MMDM", weights=weights, shard=shard))
                e.matched.verify(batch)
    write(root / f"{a.arm}-generation-complete.json", dict(complete=True, arm=a.arm, outputs=16000,
          protocol_sha256=digest(root / "protocol.json"), completed_at=now(), review_status="pending", training_eligible=False))


def qc(a):
    root = locations(a.base)["root"]
    p = validate(a.base)
    done = read(root / f"{a.arm}-generation-complete.json")
    require(done["complete"] and done["outputs"] == 16000
            and done["protocol_sha256"] == digest(root / "protocol.json"), "Generation is not finalized")
    evidence = read(root / "templates/shard-0-light-qc-v4-receipt.json")
    require(digest(a.repo / "configs/qc_policy.yaml") == evidence["parent_policy_sha256"]
            and digest(a.repo / "configs/tiling.yaml") == evidence["tiling_sha256"], "QC policy/tiling differs from Half A")
    script = root / "templates/qc_synthetics_light.py"
    for old, expected in evidence["code_sha256"].items():
        path = script if old.endswith("qc_synthetics_light.py") else a.repo / "data_prep" / Path(old).name
        require(digest(path) == expected, f"QC code differs from Half A: {path.name}")
    groups = ["corners-16000"] if a.arm == "crop" else ["resize-original-4000", "resize-additional12000"]
    receipts, total = [], Counter()
    for name in groups:
        for shard in (0, 1):
            batch = root / name
            out = root / "qc/synthetic-light-v4" / name / f"shard-{shard}"
            if out.exists() and not (out / "receipt.json").is_file():
                # Preserve an interrupted QC attempt without touching any images.
                suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                out.rename(out.with_name(out.name + ".incomplete-" + suffix))
            if not out.exists():
                env = os.environ.copy()
                env["LD_PRELOAD"] = ""
                env["LD_LIBRARY_PATH"] = str(a.base / "patch-env/lib/python3.12/site-packages/torch/lib")
                env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
                command = [str(a.base / "patch-env/bin/python"), str(script), "--repo", str(a.repo),
                           "--batch", str(batch), "--shard", str(shard), "--blur-reject-below", "10", "--out", str(out)]
                subprocess.run(command, check=True, env=env)
            receipt = read(out / "receipt.json")
            require(receipt["complete"] and receipt["protocol"] == "synthetic-light-v4"
                    and receipt["runtime"] == evidence["runtime"]
                    and receipt["effective_policy"] == evidence["effective_policy"], "QC runtime/settings differ from Half A")
            require(receipt["manifest_sha256"] == digest(batch / f"generated/shard-{shard}/generation_manifest.csv"), "QC source changed")
            for filename, expected in receipt["output_sha256"].items():
                require(digest(out / filename) == expected, "QC CSV changed")
            total.update(receipt["counts"])
            receipts.append(dict(path=str(out / "receipt.json"), sha256=digest(out / "receipt.json")))
    require(sum(total.values()) == 16000, "QC did not cover all 16,000 raw variants")
    summary = dict(complete=True, arm=a.arm, counts=dict(total), inputs=16000, receipts=receipts,
                   review_status="pending", training_eligible=False, completed_at=now())
    write(root / f"{a.arm}-qc-complete.json", summary)
    print(json.dumps(summary, indent=2))


def status(a):
    root = locations(a.base)["root"]
    validate(a.base)
    for arm, names in (("crop", ["corners-16000"]), ("resize", ["resize-original-4000", "resize-additional12000"])):
        total = 0
        for name in names:
            for shard in (0, 1):
                folder = root / name / f"generated/shard-{shard}"
                records = list((folder / "records").glob("*.json"))
                count = sum(len(read(path)["outputs"]) for path in records)
                total += count
                print(f"{arm} / {name} / shard {shard}: {count} saved variants")
                if (folder / "progress.json").exists():
                    print(json.dumps(read(folder / "progress.json"), indent=2))
        print(f"Half B {arm}: {total}/16000; finalized={(root / f'{arm}-generation-complete.json').exists()}")


def record(a):
    """Copy small provenance/selection metadata for an explicit user Git commit."""
    root = locations(a.base)["root"]
    validate(a.base)
    destination = a.repo / "alternative_models/records/half-B-fold0"
    pairs = {
        "protocol.json": root / "protocol.json",
        "source_plan.json": root / "resize-original-4000/batch_plan.json",
        "source_provenance.json": root / "resize-original-4000/references/references.json",
        "selected_source_metadata.csv": root / "resize-original-4000/references/input_manifest.csv",
        "patient_allocation.csv": root / "resize-original-4000/patient_allocation.csv",
        "B_package_manifest.json": root / "resize-original-4000/references/package_manifest.json",
        "A_synthetic_qc_configuration.json": root / "templates/shard-0-light-qc-v4-receipt.json",
    }
    for arm in ("crop", "resize"):
        for phase in ("generation", "qc"):
            name = f"{arm}-{phase}-complete.json"
            if (root / name).is_file():
                pairs[name] = root / name
    # Validate all existing targets before writing any new one.
    for name, source in pairs.items():
        target = destination / name
        require(not target.exists() or digest(target) == digest(source), f"Recorded metadata changed; inspect before replacing: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in pairs.items():
        if not (destination / name).exists():
            shutil.copyfile(source, destination / name)
    write(destination / "record_manifest.json", dict(
        files={name: digest(source) for name, source in sorted(pairs.items())},
        scope="Provenance and selection metadata only; no tile images/weights; no automatic Git operations"))
    print(f"Metadata ready for your explicit Git commit: {destination}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "validate", "status", "worker", "qc", "a-complete", "a-status", "finish-crops", "scheduler-gate", "record"))
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--arm", choices=("crop", "resize"))
    p.add_argument("--scheduler-table", type=Path)
    a = p.parse_args()
    require(a.command not in {"worker", "qc", "a-complete", "scheduler-gate"} or a.arm, "This command needs --arm")
    if a.command == "prepare": prepare(a)
    elif a.command == "validate": validate(a.base); print("Half B plan validated")
    elif a.command == "finish-crops": finish_crop_preparation(a.base); validate(a.base)
    elif a.command == "a-complete": a_complete(a.base, a.arm); print(f"Half A {a.arm} completion gate passed")
    elif a.command == "a-status": a_status(a.base)
    elif a.command == "scheduler-gate": print(json.dumps(scheduler_gate(a.scheduler_table.read_text(), a.arm), indent=2))
    else: globals()[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Half B stopped: {error}")
