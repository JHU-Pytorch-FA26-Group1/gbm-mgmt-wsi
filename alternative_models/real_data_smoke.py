"""Combine/inventory A+B, then perform a bounded real-tile ConvNeXt diagnostic.

Inventory reads metadata only. Smoke selects both classes from both halves in
both roles; it is explicitly invalid for model-quality conclusions. Full-cohort
classification remains the subsequent experiment, using all declared pool tiles.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"data_prep"))

from alternative_models.prepare import read_csv, sha, write_json


def inspect(package, repo):
    import loader
    settings_path = package/"combined_settings.json"
    if not settings_path.is_file():
        raise ValueError("Expected the combined A+B package")
    settings = json.loads(settings_path.read_text())
    if not settings.get("complete") or set(settings.get("parts", {})) != {"A", "B"}:
        raise ValueError("Incomplete or non-A+B combination")
    problems = loader.folder_problems(package)
    if problems:
        raise ValueError("; ".join(problems[:5]))
    labels_rows, folds_rows = (read_csv(repo/"manifests"/name) for name in ("patient_labels.csv", "patient_folds.csv"))
    labels = {r["patient_id"]: r for r in labels_rows}
    folds = {r["patient_id"]: r for r in folds_rows}
    if len(labels) != len(labels_rows) or len(folds) != len(folds_rows) or set(labels) != set(folds):
        raise ValueError("Canonical patient identities are duplicated or inconsistent")
    index = read_csv(package/"patient_index.csv")
    if len({r["patient_id"] for r in index}) != len(index) or {r["patient_id"] for r in index} != set(labels):
        raise ValueError("Combined index does not account for every canonical patient exactly once")
    facts, roots, evidence = {}, set(), {str(settings_path): sha(settings_path)}
    for half, part in settings["parts"].items():
        root = (package/part["path"]).resolve()
        manifest = json.loads((root/"package_manifest.json").read_text())
        if root in roots or manifest["subset"] != half or manifest["folds_sha256"] != sha(repo/"manifests/patient_folds.csv"):
            raise ValueError("Aliased halves, incorrect ownership or incompatible folds")
        roots.add(root)
        for name in ("package_manifest.json", "modeling_patches.csv"):
            evidence[str(root/name)] = sha(root/name)
    for row in index:
        patient, half = row["patient_id"], row["part"]
        if half not in settings["parts"] or labels[patient]["source"] != folds[patient]["source"]:
            raise ValueError("Invalid half or inconsistent canonical cohort")
        if labels[patient]["mgmt_status"] not in ("Methylated", "Unmethylated"):
            raise ValueError("Unknown MGMT label")
        facts[patient] = dict(half=half, source=folds[patient]["source"], fold=folds[patient]["fold"],
                              label=int(labels[patient]["mgmt_status"] == "Methylated"), tiles=0)
    identifiers, tile_counts = set(), Counter()
    with (package/"modeling_patches.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            patient, patch = row["patient_id"], row["patch_id"]
            if patch in identifiers:
                raise ValueError(f"Duplicate model tile ID: {patch}")
            identifiers.add(patch)
            if patient not in facts:
                raise ValueError(f"Unrecognized patient: {patient}")
            expected = facts[patient]
            if (row["part"], row["source"], row["fold"]) != (expected["half"], expected["source"], expected["fold"]):
                raise ValueError(f"Tile half/cohort/fold mismatch: {patch}")
            expected["tiles"] += 1
            tile_counts[expected["source"]] += 1
    evidence.update({str(package/name): sha(package/name) for name in ("modeling_patches.csv", "patient_index.csv")})
    evidence.update({str(repo/"manifests"/name): sha(repo/"manifests"/name) for name in ("patient_labels.csv", "patient_folds.csv")})
    return dict(complete=True, phase="cohort-inventory-only", package=str(package),
                tiles=sum(tile_counts.values()), tiles_by_cohort=dict(tile_counts),
                patients_with_tiles=dict(Counter(r["source"] for r in facts.values() if r["tiles"])),
                patients_without_model_tiles=sorted(p for p,r in facts.items() if not r["tiles"]),
                patients=facts, input_sha256=evidence, pixel_files_read=0)


def inventory(args):
    import combine
    if args.out.exists():
        raise ValueError("Inventory receipt exists; use a new output path")
    package = args.base/"packages/AB_qc-v5"
    if not package.exists():
        try:
            status = combine.main(["--parts", f"A={args.base}/packages/A_qc-v5", f"B={args.base}/packages/B_qc-v5",
                                   "--out", str(package), "--folds", str(args.repo/"manifests/patient_folds.csv"),
                                   "--log-dir", str(args.out.parent/"combine-logs")])
        finally:
            for handler in list(combine.log.handlers):
                handler.close()
                combine.log.removeHandler(handler)
        if status:
            raise ValueError("A+B combination failed; no partial-data override is permitted")
    settings = json.loads((package/"combined_settings.json").read_text())
    if any((package/part["path"]).resolve() != (args.base/f"packages/{half}_qc-v5").resolve()
           for half,part in settings.get("parts", {}).items()):
        raise ValueError("Existing AB_qc-v5 index points to different source packages")
    result = inspect(package, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("patients", "input_sha256")}, indent=2))


def choose(record, fold, seed):
    from alternative_models.sampling import rng
    result = {}
    for role in ("train", "validation"):
        for half in ("A", "B"):
            for label in (0, 1):
                candidates = sorted(p for p,r in record["patients"].items() if r["source"] == "TCGA"
                    and r["tiles"] > 0 and r["half"] == half and r["label"] == label
                    and (r["fold"] == str(fold)) == (role == "validation"))
                if not candidates:
                    raise ValueError(f"No {half}/{role}/label {label} patient with tiles")
                patient = rng(seed, role, half, label).choice(candidates)
                result[patient] = dict(record["patients"][patient], role=role)
    return result


def selected_rows(package, selected, count, seed):
    from alternative_models.sampling import diverse_order
    rows = {p: [] for p in selected}
    with (package/"modeling_patches.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            patient = raw["patient_id"]
            if patient in rows:
                rows[patient].append(dict(tile_id=raw["patch_id"], parent_patch_id=raw["patch_id"],
                    kind="real", patient_id=patient, label=selected[patient]["label"],
                    slide_id=raw["slide_id"], x_target=raw["x_target"], y_target=raw["y_target"], real_row=raw))
    return {p: diverse_order(r, (seed, "smoke", p))[:count] for p,r in rows.items()}


def smoke(args):
    import loader
    import torch
    from sklearn.metrics import roc_auc_score
    from alternative_models import engine
    from alternative_models.patient_metrics import summarize
    from alternative_models.sampling import coverage_bags, rng
    from alternative_models.whole_patient import backward_patient, finetune_parameters
    if args.out.exists() or not 1 <= args.tiles_per_patient <= 1024 or args.tile_chunk < 1:
        raise ValueError("New output directory, 1..1024 smoke tiles/patient and positive chunk required")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("CPU or a single CUDA device required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Run the real-data smoke in the quarter-GPU training environment")
    cohort = inspect(args.package, args.repo)
    people = choose(cohort, args.fold, args.seed)
    rows = selected_rows(args.package, people, args.tiles_per_patient, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(phase="real-data-pipeline-smoke", complete=False, valid_for_results=False,
        purpose="Engineering diagnostic, not a full-cohort or full-patient accuracy estimate",
        backbone=args.backbone, pretrained=True, class_weight=1.0, cohort=cohort,
        git_commit=engine.git_head(REPO),
        selection={p:dict(people[p], selected_tiles=len(r), tile_ids=[x["tile_id"] for x in r]) for p,r in rows.items()},
        source_sha256=engine.source_hashes(REPO), runtime=engine.runtime(torch),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(args.out/"receipt.json", record)
    pixels = None
    try:
        net, settings = engine.build(args.backbone, args.seed, pretrained=True)
        reference = args.reference_patch_id or settings["macenko_reference_patch_ids"][str(args.fold)]
        with loader.TileReader(args.package) as reader:
            normalizer, stain = loader.fit_fold_normalizer(args.package, args.fold, reference, reader)
        record["normalization"] = stain
        torch.save(net.state_dict(), args.out/"initial_pretrained_model.pt")
        record["initial_model_sha256"] = sha(args.out/"initial_pretrained_model.pt")
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        net.to(args.device)
        pixels = engine.Pixels(dict(package=str(args.package)), normalizer, args.out)
        # Cache all declared smoke tiles. Real-data reads retain owner pixel-hash checks.
        features, encoding_started = {}, time.monotonic()
        for patient, tiles in sorted(rows.items()):
            features[patient] = engine.encode_rows(net, tiles, pixels, args.device, min(args.tile_chunk, 32))
            print(f"Encoded {patient}: {len(tiles)} real tiles from half {people[patient]['half']}", flush=True)
        record["timings"] = dict(initial_encoding_seconds=time.monotonic()-encoding_started)
        targets = {p:torch.tensor([float(f["label"])], device=args.device) for p,f in people.items()}
        loss_fn = torch.nn.BCEWithLogitsLoss()
        train = sorted(p for p,r in people.items() if r["role"] == "train")
        validation = sorted(p for p,r in people.items() if r["role"] == "validation")
        for stage in ("frozen", "finetune"):
            stage_started = time.monotonic()
            net.train()
            if stage == "frozen":
                net.backbone.requires_grad_(False).eval()
                optimizer = torch.optim.Adam(net.pool.parameters(), lr=1e-4)
            else:
                encoder = finetune_parameters(net, args.backbone)
                optimizer = torch.optim.Adam([{"params":encoder, "lr":1e-6}, {"params":net.pool.parameters(), "lr":1e-5}])
            for patient in train:
                net.train()
                optimizer.zero_grad(set_to_none=True)
                if stage == "frozen":
                    net.backbone.eval()
                    logit, _ = net.pool(features[patient].unsqueeze(0).to(args.device))
                    loss = loss_fn(logit, targets[patient]); loss.backward()
                else:
                    chunks = coverage_bags(rows[patient], args.tile_chunk, args.seed, 0)
                    def load(index):
                        return torch.stack([pixels.tile(r, rng(args.seed, r["tile_id"], "smoke-augment").getrandbits(63)) for r in chunks[index]])
                    backward_patient(net, chunks, load, targets[patient], loss_fn, args.device)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise ValueError("Nonfinite real-data smoke gradients")
                optimizer.step()
            predictions = []
            net.eval()
            with torch.no_grad():
                for patient in validation:
                    emb = features[patient] if stage == "frozen" else engine.encode_rows(net, rows[patient], pixels, args.device, min(args.tile_chunk,32))
                    logit, _ = net.pool(emb.unsqueeze(0).to(args.device))
                    predictions.append(dict(patient_id=patient, label=people[patient]["label"],
                        score=torch.sigmoid(logit).item(), logit=logit.item(), fold=args.fold))
            metrics = summarize(predictions)
            metrics.update(auroc=roc_auc_score([r["label"] for r in predictions], [r["score"] for r in predictions]),
                valid_for_results=False, patient_losses=len(train), optimizer_updates=len(train),
                training_tiles=sum(len(rows[p]) for p in train), phase=stage)
            write_json(args.out/f"{stage}_patient_metrics.json", metrics)
            for prediction in predictions:
                engine.append(args.out/"development_predictions.csv", dict(stage=stage, **prediction))
            engine.save_model(args.out/f"{stage}.pt", net)
            record["timings"][stage+"_training_and_scoring_seconds"] = time.monotonic()-stage_started
            print(f"{stage}: one pass over {len(train)} training patients; {len(validation)} diagnostic predictions", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            record.update(peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved(args.device)/2**30,
                visible_memory_GiB=torch.cuda.get_device_properties(args.device).total_memory/2**30)
        record.update(complete=True, elapsed_seconds=time.monotonic()-started)
        write_json(args.out/"receipt.json", record)
        print(json.dumps({k:record[k] for k in ("complete", "valid_for_results", "elapsed_seconds", "peak_allocated_GiB", "peak_reserved_GiB") if k in record}, indent=2))
    except Exception as exc:
        record.update(failure=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        write_json(args.out/"receipt.json", record)
        raise
    finally:
        if pixels is not None:
            pixels.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--base", type=Path, default=Path("/workspace/persist/crivas4-microscopy"))
    run = commands.add_parser("smoke")
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--backbone", choices=("convnext_v2_t", "swin_v2_t"), default="convnext_v2_t")
    run.add_argument("--fold", type=int, choices=range(5), default=0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--tiles-per-patient", type=int, default=129)
    run.add_argument("--tile-chunk", type=int, default=128)
    run.add_argument("--reference-patch-id")
    run.add_argument("--device", default="cuda:0")
    for subparser in (inv, run):
        subparser.add_argument("--repo", type=Path, default=REPO)
        subparser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (inventory if args.command == "inventory" else smoke)(args)


if __name__ == "__main__":
    main()
