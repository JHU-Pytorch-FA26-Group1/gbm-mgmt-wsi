"""Prepare paired patient pools from verified real packages and accepted synthetic CSVs."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from alternative_models.sampling import select_pools


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes its artifact root: {relative}")
    return path


def prepare(args):
    import loader
    package, repo = args.package.resolve(), args.repo.resolve()
    if args.out.exists():
        raise ValueError("Output already exists; refusing overwrite")
    evidence_files = {}
    if getattr(args, "require_combined", False) and not getattr(args, "allow_single_package_smoke", False):
        settings_path = package / "combined_settings.json"
        if not settings_path.is_file():
            raise ValueError("Classification preparation requires the combined A+B input, not a single half")
        settings = json.loads(settings_path.read_text())
        if not settings.get("complete") or set(settings.get("parts", {})) != {"A", "B"}:
            raise ValueError("A complete combination with distinct A and B parts is required")
        paths = [settings_path, package / "patient_index.csv"]
        roots = set()
        for half, part in settings["parts"].items():
            root = (package / part["path"]).resolve()
            if root in roots or json.loads((root / "package_manifest.json").read_text()).get("subset") != half:
                raise ValueError("Combined A+B must reference distinct, correctly identified halves")
            roots.add(root)
            paths += [root / "package_manifest.json", root / "modeling_patches.csv"]
        evidence_files = {str(path): sha(path) for path in paths}
    folds_path, labels_path = repo / "manifests/patient_folds.csv", repo / "manifests/patient_labels.csv"
    fold_rows, label_rows = read_csv(folds_path), read_csv(labels_path)
    folds = {r["patient_id"]: (r["source"], r["fold"]) for r in fold_rows}
    labels = {r["patient_id"]: int(r["mgmt_status"] == "Methylated") for r in label_rows}
    if len(folds) != len(fold_rows) or len(labels) != len(label_rows) or set(folds) != set(labels):
        raise ValueError("Canonical labels/folds contain duplicates or disagree")
    if any(r["mgmt_status"] not in ("Methylated", "Unmethylated") for r in label_rows):
        raise ValueError("Unknown canonical MGMT status")
    if any(r["source"] != folds[r["patient_id"]][0] for r in label_rows):
        raise ValueError("Label and fold source cohorts disagree")
    table = loader.load_modeling(package)
    if evidence_files:
        index = read_csv(package / "patient_index.csv")
        if len({r["patient_id"] for r in index}) != len(index) or {r["patient_id"] for r in index} != set(folds):
            raise ValueError("Combined patient index must account for every canonical patient exactly once")
    real, train, validation = {}, defaultdict(list), defaultdict(list)
    selected = None
    roi_counts = None
    if getattr(args, "roi_scores", None):
        from alternative_models.roi_pool import select_roi
        if args.synthetic_inputs or args.cap is not None:
            raise ValueError("ROI experiment requires all eligible real tiles")
        selected, roi_evidence, roi_counts = select_roi(table, args.roi_scores)
        evidence_files.update(roi_evidence)
    for values in table.itertuples(index=False, name=None):
        record = dict(zip(table.columns, values))
        patient, patch = record["patient_id"], record["patch_id"]
        if patient not in folds or folds[patient] != (record["source"], str(record["fold"])):
            raise ValueError(f"Package/canonical fold mismatch: {patch}")
        if record["source"] != "TCGA":
            continue  # CPTAC is not read, normalized, encoded or scored by this CV runner.
        if patch in real:
            raise ValueError(f"Duplicate real patch ID: {patch}")
        if selected is not None and patch not in selected:
            continue
        row = dict(tile_id=patch, parent_patch_id=patch, patient_id=patient,
                   slide_id=record["slide_id"], kind="real", label=labels[patient],
                   x_target=record["x_target"], y_target=record["y_target"],
                   real_row=record)
        real[patch] = row
        (validation if str(record["fold"]) == str(args.fold) else train)[patient].append(row)
    if (not train or not validation or {labels[p] for p in train} != {0, 1}
            or {labels[p] for p in validation} != {0, 1}):
        raise ValueError("Need training and real validation patients of both classes")
    specifications = json.loads(args.synthetic_inputs.read_text()) if args.synthetic_inputs else []
    variants, seen, evidence = defaultdict(lambda: defaultdict(list)), set(), []
    for spec in specifications:
        arm = spec["design"]
        if arm not in ("resize", "crop"):
            raise ValueError("Synthetic design must be resize or crop")
        accepted, receipt = Path(spec["accepted_csv"]).resolve(), Path(spec["receipt"]).resolve()
        root = Path(spec["image_root"]).resolve()
        facts = json.loads(receipt.read_text())
        if not facts.get("complete") or sha(accepted) != facts["output_sha256"].get(accepted.name):
            raise ValueError("Incomplete QC receipt or changed accepted CSV")
        if facts.get("folds_sha256") != sha(folds_path):
            raise ValueError("QC fold map differs from the current canonical map")
        namespace = spec["namespace"]  # e.g. half-A; distinguishes ref IDs in independent batches.
        for record in read_csv(accepted):
            parent, patient = record["patch_id"], record["patient_id"]
            if record["pixel_qc_decision"] != "accept":
                raise ValueError("Nonaccepted row in accepted CSV")
            if parent not in real or patient not in train or real[parent]["patient_id"] != patient:
                raise ValueError(f"Synthetic parent unavailable or held out: {parent}")
            if (record["source"] != "TCGA" or str(record["held_out_fold"]) != str(args.fold)
                    or str(record["assigned_fold"]) != folds[patient][1]
                    or record["slide_id"] != real[parent]["slide_id"]):
                raise ValueError("Synthetic source/slide/fold lineage mismatch")
            expected_label = "methylated" if labels[patient] else "unmethylated"
            if record["mgmt_patient_label"].lower() != expected_label:
                raise ValueError("Inherited MGMT status differs from canonical label")
            if record["pixel_sha256"] != real[parent]["real_row"]["pixel_sha256"]:
                raise ValueError("Source pixel hash differs from the real package")
            tile_id = f"{arm}:{namespace}:{record['variant_id']}"
            if tile_id in seen:
                raise ValueError(f"Duplicate variant: {tile_id}")
            seen.add(tile_id)
            image = inside(root, record["classifier_path"])
            if not image.is_file():
                raise ValueError(f"Missing classifier image: {image}")
            row = {k: v for k, v in real[parent].items() if k != "real_row"}
            row.update(tile_id=tile_id, kind="synthetic", design=arm, path=str(image),
                       image_sha256=record["classifier_sha256"],
                       original_variant_id=record["variant_id"])
            variants[arm][patient].append(row)
        evidence.append(dict(spec=spec, accepted_sha256=sha(accepted), receipt_sha256=sha(receipt),
                             effective_policy=facts.get("effective_policy")))
    if specifications and not variants:
        raise ValueError("No accepted variants were provided")
    pools = {arm: {} for arm in ["real", *sorted(variants)]}
    counts = []
    for patient in sorted(train):
        selected = select_pools(train[patient],
                                {arm: variants[arm][patient] for arm in variants}, args.cap, args.seed)
        for arm, rows in selected.items():
            pools[arm][patient] = rows
            synthetic = sum(r["kind"] == "synthetic" for r in rows)
            counts.append(dict(arm=arm, patient_id=patient, real=len(rows)-synthetic,
                               synthetic=synthetic, total=len(rows), fraction=synthetic/len(rows)))
    plan = dict(protocol="alternative-attention-v1", fold=args.fold, seed=args.seed,
                cap=args.cap, package=str(package), repo=str(repo),
                package_evidence=evidence_files,
                smoke_only=getattr(args, "allow_single_package_smoke", False),
                population=dict(training_patients=len(train), validation_patients=len(validation),
                    tcga_patients_with_tiles=len(train)+len(validation),
                    tcga_patients_without_model_tiles=sorted(p for p, (source, _) in folds.items()
                        if source == "TCGA" and p not in train and p not in validation)),
                modeling_sha256=sha(package / "modeling_patches.csv"),
                folds_sha256=sha(folds_path), labels_sha256=sha(labels_path),
                synthetic_inputs=evidence, pools=pools, validation=dict(validation),
                class_weight=sum(labels[p] == 0 for p in train)/sum(labels[p] == 1 for p in train),
                source_families={arm: sorted({r["parent_patch_id"] for rows in patients.values()
                                 for r in rows if r["kind"] == "synthetic"})
                                 for arm, patients in pools.items() if arm != "real"},
                notes=["Pixel QC acceptance and inherited MGMT labels are not biological certification",
                       "All accepted variants retained; actual synthetic fractions may differ by design",
                       "All-tiles sets the real-only budget to all available tiles; synthetic arms still replace real tiles to match that budget",
                       "Frozen features use deterministic preprocessing; fine-tuning adds shared flips/quarter-turns"])
    if "crop" in plan["source_families"] and "resize" in plan["source_families"]:
        plan["same_accepted_source_families"] = plan["source_families"]["crop"] == plan["source_families"]["resize"]
    args.out.mkdir(parents=True, exist_ok=False)
    if roi_counts is not None:
        plan["roi_selection"] = {"minimum_tumor_fraction": 0.5, "eligible_by_patient": dict(roi_counts)}
    with (args.out / "plan.json").open("w") as stream:
        json.dump(plan, stream, allow_nan=False, separators=(",", ":"))

    with (args.out / "patient_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    print(f"Prepared {len(train)} training / {len(validation)} validation TCGA patients; arms {list(pools)}")
    print(f"Review {args.out / 'patient_counts.csv'} before encoding or training")
