"""Exploratory, blinded GBM ROI overlays from exact neighboring A+B package tiles.

This never writes training eligibility or a production tumor filter. Original
checkpoint MPP/reference are unknown; the preview records these limitations.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import importlib.metadata
import io
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import time

from alternative_models.roi_checkpoint import session, sha, write_json

TILE = 224
SIZE = 1024
GRID = 5
MARGIN = (GRID * TILE - SIZE) // 2


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or nonlocal package payload: {relative}")
    return path


def package_info(root, half, fold_path):
    root = root.resolve()
    manifest_path = root / "package_manifest.json"
    info = json.loads(manifest_path.read_text())
    if (info.get("subset") != half or info.get("complete") is not True
            or info.get("kind") != "package" or info.get("tiling_version") != "tiling-v7"
            or info.get("folds_sha256") != sha(fold_path)):
        raise ValueError(f"Wrong/incomplete package or fold map: {half}")
    for name in ("modeling_patches.csv", "patient_index.csv", "export_settings.json"):
        item = info["files"][name]
        path = contained(root, name)
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Package metadata changed: {half}/{name}")
    return info


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_rows(base, repo, excluded_fold, patients_per_half, seed,
                regions_per_patient=1, availability_path=None):
    if excluded_fold not in range(5):
        raise ValueError("An outer fold from 0 through 4 must be excluded")
    fold_path = repo / "manifests/patient_folds.csv"
    records = read_csv(fold_path)
    folds = {r["patient_id"]: r for r in records}
    if len(folds) != len(records):
        raise ValueError("Duplicate canonical patients")
    if any(r["source"] == "TCGA" and r["fold"] not in {str(i) for i in range(5)} for r in records):
        raise ValueError("Invalid canonical TCGA fold")
    evidence = {str(fold_path): sha(fold_path)}
    groups, owners, packages, shared = {}, {}, {}, None
    availability, shortages = {}, []
    for half in ("A", "B"):
        root = (base / f"packages/{half}_qc-v5").resolve()
        info = package_info(root, half, fold_path)
        identity = {k:info[k] for k in ("tiling_version","config_sha256","code_version","policy_sha256","folds_sha256")}
        if shared is not None and identity != shared:
            raise ValueError("A/B tiling, code, QC policy or fold identities differ")
        shared = identity
        packages[half] = dict(path=str(root), manifest_sha256=sha(root/"package_manifest.json"),
                              modeling_sha256=sha(root/"modeling_patches.csv"))
        eligible = []
        for row in read_csv(root/"patient_index.csv"):
            patient = row["patient_id"]
            if patient not in folds or patient in owners:
                raise ValueError(f"Unknown or duplicated A/B patient: {patient}")
            owners[patient] = half
            canonical = folds[patient]
            if (canonical["source"] == "TCGA" and canonical["fold"] != str(excluded_fold)
                    and int(row["model_tiles"]) > 0):
                eligible.append(patient)
        # Inspect geometry before sampling. The export groups CSV rows by slide;
        # only one slide's coordinates need to live in memory at a time.
        local = {patient: [] for patient in eligible}
        counts = dict.fromkeys(eligible, 0)
        seen_ids = set()
        seen_slides = set()
        print(f"{half}: scanning training-patient coordinates (metadata only)",flush=True)
        with (root/"modeling_patches.csv").open(newline="") as handle:
            for slide_id, rows in itertools.groupby(csv.DictReader(handle),key=lambda r:r["slide_id"]):
                if slide_id in seen_slides:
                    raise ValueError("Package CSV must group all rows of each slide together")
                seen_slides.add(slide_id)
                positions, patient, slide_patient = {}, None, None
                for row in rows:
                    if slide_patient is not None and slide_patient != row["patient_id"]:
                        raise ValueError("A slide mixes patient identities")
                    slide_patient = row["patient_id"]
                    if row["patient_id"] not in local:
                        continue
                    patient = row["patient_id"]
                    if (row["source"] != "TCGA" or row["fold"] != folds[patient]["fold"]
                            or row["fold"] == str(excluded_fold) or row["stored"] != "True"):
                        raise ValueError("Held-out/non-TCGA/misassigned/nonportable selected tile")
                    x, y = int(row["x_target"]), int(row["y_target"])
                    if x < 0 or y < 0 or x % TILE or y % TILE or (x,y) in positions or row["patch_id"] in seen_ids:
                        raise ValueError("Off-grid/duplicated selected tile")
                    seen_ids.add(row["patch_id"])
                    positions[x,y] = {k:row[k] for k in ("patient_id", "slide_id", "patch_id", "h5", "h5_index", "pixel_sha256")}
                if patient is None:
                    continue
                complete = neighborhoods({slide_id:positions})
                counts[patient] += len(complete)
                for choice in spaced_choices(complete,regions_per_patient,seed,patient):
                    local[patient].append((choice,{key:positions[key] for key in choice[3]}))
        candidates = {}
        for patient, options in local.items():
            chosen_regions = spaced_choices([x[0] for x in options],regions_per_patient,seed,patient)
            found = len(chosen_regions)
            availability[patient] = dict(half=half,complete_5x5_neighborhoods=counts[patient],
                separated_regions_found=min(found,regions_per_patient),selected=False,
                reason="eligible_not_sampled" if found == regions_per_patient else "insufficient_contiguous_regions")
            if found == regions_per_patient:
                by_position = {item[0][:3]:item for item in options}
                candidates[patient] = [by_position[c[:3]] for c in chosen_regions]
        def rank(patient):
            return hashlib.sha256(f"{seed}|{half}|{patient}".encode()).hexdigest()
        selected_patients = sorted(candidates,key=rank)[:patients_per_half]
        print(f"{half}: {len(candidates)}/{len(eligible)} training patients have {regions_per_patient} intact separated regions; selected {len(selected_patients)}",flush=True)
        if not selected_patients:
            shortages.append(f"{half}: no training patients have {regions_per_patient} intact separated regions")
        for patient in selected_patients:
            slides = defaultdict(dict)
            for choice, positions in candidates[patient]:
                slides[choice[0]].update(positions)
            groups[patient] = dict(half=half,root=root,slides=dict(slides),
                plan=[item[0] for item in candidates[patient]])
            availability[patient].update(selected=True,reason="selected")
        del info
    if set(owners) != set(folds):
        raise ValueError("A+B do not account for all canonical patients")
    report = dict(phase="geometry-preflight",complete=not shortages,pixels_read=0,predictions_run=0,
        selection_basis="source coordinates only; no pixels, tumor scores or MGMT labels used",
        requested_patients_per_half=patients_per_half,required_regions_per_patient=regions_per_patient,
        seed=seed,exclude_fold=excluded_fold,patients=availability,
        limitation="Feasibility preview restricted to reconstructable tissue; not representative accuracy evaluation",
        separated_region_count_note="Capped at the requested regions per patient; not a count of all disjoint regions")
    if availability_path is not None:
        availability_path.parent.mkdir(parents=True,exist_ok=True)
        if availability_path.exists():
            raise ValueError("Availability report exists; choose a fresh output path")
        write_json(availability_path,report)
        print(f"Region availability: {availability_path}",flush=True)
    if shortages:
        raise ValueError("; ".join(shortages) + "; see the availability report. No pixels padded or enlarged.")
    return groups, dict(fold_sha256=evidence[str(fold_path)], packages=packages,region_availability=report)


def neighborhoods(slides):
    """Only complete, truly adjacent 5x5 neighborhoods; never fill missing tissue."""
    choices = []
    for slide_id, positions in sorted(slides.items()):
        for x,y in sorted(positions):
            keys = [(x+dx*TILE, y+dy*TILE) for dy in range(GRID) for dx in range(GRID)]
            if all(key in positions for key in keys):
                choices.append((slide_id, x, y, keys))
    return choices


def spaced_choices(choices, limit, seed, patient):
    ordered = sorted(choices, key=lambda c: hashlib.sha256(f"{seed}|{patient}|{c[:3]}".encode()).hexdigest())
    best = []
    # A central first draw can block two valid edge regions. Deterministic
    # spatial fallbacks avoid mistaking that greedy failure for missing pixels.
    alternatives = (ordered,sorted(choices,key=lambda c:(c[0],c[1],c[2])),
                    sorted(choices,key=lambda c:(c[0],c[2],c[1])))
    for order in alternatives:
        result = []
        for item in order:
            if all(item[0] != other[0] or abs(item[1]-other[1]) >= GRID*TILE
                   or abs(item[2]-other[2]) >= GRID*TILE for other in result):
                result.append(item)
                if len(result) == limit:
                    return result
        if len(result) > len(best):
            best = result
    return best


def read_region(root, positions, choice):
    import h5py
    import numpy as np
    from PIL import Image
    slide_id, x, y, keys = choice
    canvas = np.empty((GRID*TILE, GRID*TILE, 3), dtype=np.uint8)
    evidence = []
    for px, py in keys:
        row = positions[px,py]
        if row["slide_id"] != slide_id:
            raise ValueError("A reconstructed region mixes slides")
        path = contained(root, row["h5"])
        with h5py.File(path, "r") as handle:
            if (handle.attrs.get("patient_id") != row["patient_id"] or handle.attrs.get("slide_id") != slide_id
                    or int(handle.attrs.get("tile_px",0)) != TILE
                    or abs(float(handle.attrs.get("target_mpp",0))-0.5) > 1e-9):
                raise ValueError("HDF5 identity, pixel size or physical resolution differs")
            i = int(row["h5_index"])
            if i < 0:
                raise ValueError("Negative HDF5 tile index")
            for name, expected in (("patch_id",row["patch_id"]),("pixel_sha256",row["pixel_sha256"])):
                value = handle[name][i]
                if isinstance(value, bytes):
                    value = value.decode()
                if value != expected:
                    raise ValueError("HDF5 tile identity/hash differs from metadata")
            rgb = np.asarray(Image.open(io.BytesIO(handle["tiles"][i].tobytes())).convert("RGB"))
        if rgb.shape != (TILE,TILE,3) or hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
            raise ValueError("Corrupt source pixels; reconstruction stopped")
        canvas[py-y:py-y+TILE, px-x:px-x+TILE] = rgb
        evidence.append(dict(row, x_target=px, y_target=py))
    return canvas[MARGIN:MARGIN+SIZE,MARGIN:MARGIN+SIZE].copy(), evidence


def inference_tensor(rgb):
    """Match MONAI 1.3.2 PILReader's default spatial transpose; no /255 scaling."""
    import numpy as np
    if rgb.shape != (SIZE,SIZE,3) or rgb.dtype != np.uint8:
        raise ValueError("Expected native 1024x1024 uint8 RGB region")
    return np.ascontiguousarray(rgb.transpose(2,1,0)[None], dtype=np.float32)


def output_probability(logits):
    import numpy as np
    if logits.shape != (1,1,SIZE,SIZE) or not np.isfinite(logits).all():
        raise ValueError("Unexpected/nonfinite ONNX logits")
    # Undo the PILReader spatial transpose so overlays align with source pixels.
    value = logits[0,0].T
    return np.exp(-np.logaddexp(0., -value))


def preview(args):
    import numpy as np
    from PIL import Image
    import torchstain
    if args.out.exists() or not 1 <= args.patients_per_half <= 12 or not 1 <= args.regions_per_patient <= 4:
        raise ValueError("New output directory required; at most 12 patients/half and 4 regions/patient")
    groups, provenance = select_rows(args.base,args.repo,args.exclude_fold,args.patients_per_half,args.seed,
        args.regions_per_patient,args.out.with_name(args.out.name+"-availability.json"))
    plan = {patient:group["plan"] for patient,group in groups.items()}
    runtime = session(args.checkpoint,args.threads)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".roi-preview-",dir=args.out.parent))
    started = time.monotonic()
    try:
        first_patient = sorted(plan)[0]
        first_choice = plan[first_patient][0]
        first = groups[first_patient]
        reference, _ = read_region(first["root"],first["slides"][first_choice[0]],first_choice)
        normalizer = torchstain.normalizers.ReinhardNormalizer(method="modified",backend="numpy")
        normalizer.fit(reference)
        if not np.isfinite(normalizer.target_stds).all() or normalizer.target_stds[0] <= 0:
            raise ValueError("Degenerate provisional stain reference")
        Image.fromarray(reference).save(stage/"provisional_training_reference.png")
        records, cards = [], []
        write_json(stage/"region_availability.json",provenance["region_availability"])
        for patient, choices in sorted(plan.items()):
            group = groups[patient]
            for choice in choices:
                region_id = f"region-{len(records)+1:03d}"
                folder = stage/region_id
                folder.mkdir()
                rgb, source_rows = read_region(group["root"],group["slides"][choice[0]],choice)
                normalized = normalizer.normalize(I=rgb)
                tensor = inference_tensor(normalized)
                probability = output_probability(runtime.run(None,{runtime.get_inputs()[0].name:tensor})[0])
                mask = probability > 0.5
                overlay = rgb.copy()
                overlay[mask] = np.round(rgb[mask]*0.55 + np.array([255,0,0])*0.45).astype(np.uint8)
                Image.fromarray(rgb).save(folder/"original.png")
                Image.fromarray(normalized).save(folder/"normalized.png")
                Image.fromarray((mask*255).astype(np.uint8)).save(folder/"mask.png")
                Image.fromarray(overlay).save(folder/"overlay.png")
                np.save(folder/"tumor_probability.npy",probability.astype(np.float32),allow_pickle=False)
                metadata = dict(region_id=region_id,patient_id=patient,half=group["half"],slide_id=choice[0],
                    x_target=choice[1]+MARGIN,y_target=choice[2]+MARGIN,mpp=0.5,
                    input_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),sources=source_rows,
                    predicted_tumor_fraction=float(mask.mean()),pixel_probability_cutoff=0.5,
                    reviewer_decision="",reviewer_comments="")
                write_json(folder/"provenance.json",metadata)
                records.append(metadata)
                title = html.escape(f"{region_id} | {patient} | half {group['half']} | predicted tumor {mask.mean():.1%}")
                panels = "".join(f'<figure><img loading="lazy" src="{region_id}/{name}.png"><figcaption>{label}</figcaption></figure>'
                    for name,label in (("original","Original"),("normalized","Normalized model input"),("overlay","Predicted tumor in red"),("mask","Binary prediction")))
                cards.append(f"<section><h2>{title}</h2><div class=panels>{panels}</div></section>")
                print(f"Predicted {region_id}: {patient}, half {group['half']}",flush=True)
        fields = ("region_id","patient_id","half","slide_id","predicted_tumor_fraction","reviewer_decision","reviewer_comments")
        with (stage/"review.csv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
            writer.writeheader();writer.writerows(records)
        banner = ("Exploratory ROI review. Red indicates predicted tumor, not ground truth. "
            "Original checkpoint training resolution and stain reference are unverified. "
            "Our inputs are reconstructed at 0.5 microns/pixel without resizing or filling gaps. "
            "Please review missed tumor as well as falsely marked tumor. "
            "Record usable / needs_correction / unusable in review.csv with comments. "
            "No MGMT labels are shown; no training eligibility has been changed.")
        (stage/"review.html").write_text('<!doctype html><meta charset="utf-8"><title>Exploratory GBM ROI review</title>'
            '<style>body{font:16px sans-serif;margin:24px;background:#fafafa}section{margin:32px 0}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}figcaption{padding:8px}p{max-width:1100px}</style>'
            '<h1>GBM tumor segmentation — exploratory review</h1><p>'+html.escape(banner)+'</p>'+"".join(cards))
        versions = {name:importlib.metadata.version(name) for name in ("numpy","Pillow","h5py","torchstain","onnxruntime")}
        write_json(stage/"receipt.json",dict(complete=True,phase="exploratory-ROI-preview",valid_for_results=False,
            production_filter_enabled=False,checkpoint_sha256=sha(args.checkpoint),upstream_training_mpp_verified=False,
            upstream_training_patient_overlap_verified=False,
            upstream_stain_reference_available=False,used_mpp=0.5,normalized_input_range="uint8 0..255 converted to float32",
            normalization="torchstain 1.3.0 modified Reinhard, NumPy backend; not the authors' Torch backend",
            reference_region_id="region-001",provisional_reference_pixel_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            image_transpose="RGB HWC -> CWH, matching MONAI PILReader; output transposed back",
            selection=dict(exclude_fold=args.exclude_fold,cohort="TCGA training only",seed=args.seed,
                patients_per_half=args.patients_per_half,regions_per_patient=args.regions_per_patient,
                neighboring_tiles_per_region=25,crop_from_1120=MARGIN,no_padding=True,no_resizing=True),
            source_metadata=provenance,regions=len(records),versions=versions,elapsed_seconds=time.monotonic()-started,
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ("roi_preview.py","roi_checkpoint.py")}))
        if args.out.exists():
            raise ValueError("Output appeared during preview; refusing replacement")
        stage.rename(args.out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Exploratory review: {args.out/'review.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--exclude-fold",type=int,choices=range(5),default=0)
    parser.add_argument("--patients-per-half",type=int,default=4)
    parser.add_argument("--regions-per-patient",type=int,default=2)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--threads",type=int,default=2)
    preview(parser.parse_args())


if __name__ == "__main__":
    main()
