"""Frozen features, then chunked whole-patient fine-tuning.

FP32, Adam, no scheduler, patient-level weighted BCE; rates/cycle budgets are explicit CLI inputs.
Current owner backbone/pooling implementations are imported read-only and their hashes recorded.
"""
import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from alternative_models.prepare import sha, write_json
from alternative_models.sampling import coverage_bags, rng
from alternative_models.whole_patient import backward_patient, finetune_parameters, Plateau
from alternative_models.patient_metrics import summarize


def source_hashes(repo):
    paths = [*Path(repo).glob("alternative_models/*.py"),
             *Path(repo).glob("training/*.py"), *Path(repo).glob("training/backbones/*.py"),
             *Path(repo).glob("configs/backbones/*.yaml"), Path(repo)/"configs/train.yaml",
             Path(repo)/"data_prep/loader.py", Path(repo)/"data_prep/common.py"]
    return {str(p.relative_to(repo)): sha(p) for p in sorted(paths)}


def runtime(torch):
    versions = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "tiatoolbox", "pillow", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), versions=versions, cuda=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def git_head(repo):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def current_plan(path):
    plan = json.loads(Path(path).read_text())
    repo = Path(plan["repo"])
    if repo.resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError("Plan belongs to a different checkout; run from its original reviewed code")
    if plan["protocol"] != "alternative-attention-v1":
        raise ValueError("Unknown plan protocol")
    for filename, expected in plan.get("package_evidence", {}).items():
        if sha(filename) != expected:
            raise ValueError(f"Combined package input changed: {filename}")
    for path, expected in [(Path(plan["package"])/"modeling_patches.csv", plan["modeling_sha256"]),
                           (repo/"manifests/patient_folds.csv", plan["folds_sha256"]),
                           (repo/"manifests/patient_labels.csv", plan["labels_sha256"])]:
        if sha(path) != expected:
            raise ValueError(f"Plan input changed: {path}")
    for evidence in plan["synthetic_inputs"]:
        for key, field in (("accepted_csv", "accepted_sha256"), ("receipt", "receipt_sha256")):
            if sha(evidence["spec"][key]) != evidence[field]:
                raise ValueError("Synthetic QC evidence changed since preparation")
    return plan


def build(backbone, seed, pretrained):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from training import config, models
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    settings, _ = config.load_config(backbone=backbone)
    model = models.build_model(settings, backbone, pretrained=pretrained)
    return model, settings


def normalizer_from(record):
    import numpy as np
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    normalizer = MacenkoNormalizer()
    normalizer.stain_matrix_target = np.asarray(record["stain_matrix_target"], dtype=np.float64)
    normalizer.maxC_target = np.asarray(record["maxC_target"], dtype=np.float64)
    return normalizer


class Pixels:
    def __init__(self, plan, normalizer, out):
        import loader
        self.reader = loader.TileReader(plan["package"])
        self.normalizer, self.log = normalizer, Path(out)/"normalization_failures.csv"

    def tile(self, row, augment_seed=None):
        import loader
        import numpy as np
        from PIL import Image
        from training.data import augment, to_tensor
        if row["kind"] == "real":
            image = self.reader.read(row["real_row"])
        else:
            if sha(row["path"]) != row["image_sha256"]:
                raise ValueError(f"Changed synthetic image: {row['tile_id']}")
            with Image.open(row["path"]) as handle:
                if handle.mode != "RGB" or handle.size != (224, 224):
                    raise ValueError("Expected the existing RGB 224-pixel classifier copy")
                image = np.array(handle)
        image = loader.normalize_tile(self.normalizer, image, row["tile_id"], self.log)
        if augment_seed is not None:
            image = augment(image, np.random.default_rng(augment_seed))
        return to_tensor(image)

    def close(self):
        self.reader.close()


def encode_rows(model, rows, pixels, device, chunk):
    import torch
    if chunk < 1:
        raise ValueError("Positive encoding chunk required")
    pieces = []
    model.backbone.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            images = torch.stack([pixels.tile(r) for r in rows[start:start+chunk]]).to(device)
            features = model.encode(images)
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite backbone features")
            pieces.append(features.cpu())
    return torch.cat(pieces)


def cache(args):
    import loader
    import numpy as np
    import torch
    plan = current_plan(args.plan)
    if args.out.exists():
        raise ValueError("Cache directory already exists; refusing overwrite")
    model, _ = build(args.backbone, plan["seed"], pretrained=True)
    args.out.mkdir(parents=True, exist_ok=False)
    # Cache immutable initial weights, including the common attention initialization.
    torch.save(model.state_dict(), args.out/"initial_model.pt")
    reader = loader.TileReader(plan["package"])
    try:
        normalizer, stain = loader.fit_fold_normalizer(plan["package"], plan["fold"],
                                                       args.reference_patch_id, reader)
    finally:
        reader.close()
    metadata = dict(protocol="alternative-feature-cache-v1", complete=False,
                    plan=str(args.plan.resolve()), plan_sha256=sha(args.plan), backbone=args.backbone,
                    seed=plan["seed"], normalization=stain, preprocessing="Macenko then ImageNet tensor; no JPEG/crop/jitter",
                    source_sha256=source_hashes(Path(plan["repo"])), runtime=runtime(torch), patients={})
    metadata["git_commit"] = git_head(plan["repo"])
    write_json(args.out/"cache.json", metadata)
    model.to(args.device).eval().requires_grad_(False)
    pixels = Pixels(plan, normalizer, args.out)
    union = {}
    for patients in [plan["validation"], *plan["pools"].values()]:
        for patient, rows in patients.items():
            mine = union.setdefault(patient, {})
            for row in rows:
                if row["tile_id"] in mine and mine[row["tile_id"]] != row:
                    raise ValueError("Same tile ID with inconsistent content")
                mine[row["tile_id"]] = row
    try:
        for index, (patient, mapping) in enumerate(sorted(union.items())):
            rows = sorted(mapping.values(), key=lambda r: r["tile_id"])
            features = encode_rows(model, rows, pixels, args.device, args.chunk)
            filename = f"patient-{index:05d}.npz"
            np.savez(args.out/filename, tile_ids=np.asarray([r["tile_id"] for r in rows]),
                     embeddings=features.numpy())
            metadata["patients"][patient] = dict(file=filename, sha256=sha(args.out/filename), tiles=len(rows))
            write_json(args.out/"cache.json", metadata)
            print(f"Cached {index+1}/{len(union)} patients: {patient}, {len(rows)} tiles", flush=True)
    finally:
        pixels.close()
    metadata["initial_model_sha256"] = sha(args.out/"initial_model.pt")
    metadata["complete"] = True
    write_json(args.out/"cache.json", metadata)


class Features:
    def __init__(self, folder, plan_path, plan):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder/"cache.json").read_text())
        record = self.metadata
        if (not record["complete"] or record["plan_sha256"] != sha(plan_path)
                or record["source_sha256"] != source_hashes(Path(plan["repo"]))
                or record["initial_model_sha256"] != sha(self.folder/"initial_model.pt")):
            raise ValueError("Incomplete/stale cache or changed initial weights/source code")

    def get(self, patient, rows):
        import numpy as np
        import torch
        record = self.metadata["patients"][patient]
        path = self.folder/record["file"]
        if sha(path) != record["sha256"]:
            raise ValueError(f"Corrupt feature cache: {patient}")
        with np.load(path, allow_pickle=False) as arrays:
            index = {str(tile): i for i, tile in enumerate(arrays["tile_ids"])}
            result = arrays["embeddings"][[index[r["tile_id"]] for r in rows]].copy()
        return torch.from_numpy(result)


def predict(model, plan, features, pixels, args, frozen):
    import torch
    from sklearn.metrics import roc_auc_score
    model.eval()
    rows, losses = [], []
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(plan["class_weight"], device=args.device))
    with torch.no_grad():
        for patient, tiles in sorted(plan["validation"].items()):
            embeddings = (features.get(patient, tiles) if frozen else
                          encode_rows(model, tiles, pixels, args.device, args.chunk))
            logit, _ = model.pool(embeddings.unsqueeze(0).to(args.device))
            label = tiles[0]["label"]
            loss = loss_fn(logit, torch.tensor([float(label)], device=args.device))
            rows.append(dict(patient_id=patient, label=label, logit=logit.item(),
                             score=torch.sigmoid(logit).item(), fold=plan["fold"]))
            losses.append(loss.item())
    if not all(torch.isfinite(torch.tensor(r["logit"])) for r in rows):
        raise ValueError("Nonfinite validation predictions")
    auc = roc_auc_score([r["label"] for r in rows], [r["score"] for r in rows]) if len({r["label"] for r in rows}) == 2 else None
    return sum(losses)/len(losses), auc, rows


def save_model(path, model):
    import os
    import torch
    temporary = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def append(path, row):
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    import torch
    plan = current_plan(args.plan)
    if args.arm not in plan["pools"] or args.out.exists():
        raise ValueError("Unknown arm or output already exists")
    args.encoder_lr = getattr(args, "encoder_lr", args.finetune_lr / 10)
    args.min_delta = getattr(args, "min_delta", .0001)
    args.smoke = getattr(args, "smoke", False) or plan.get("smoke_only", False)
    if (not all(math.isfinite(v) and v > 0 for v in (args.frozen_lr, args.finetune_lr, args.encoder_lr))
            or min(args.frozen_cycles, args.finetune_cycles, args.patience, args.bag_size, args.chunk) < 1):
        raise ValueError("Explicit positive rates, cycle counts, patience and bag size required")
    if args.frozen_cycles + args.finetune_cycles > 200:
        raise ValueError("The combined frozen/fine-tuning budget must not exceed 200 epochs")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError("Finite nonnegative min_delta required")
    features = Features(args.cache, args.plan, plan)
    model, _ = build(features.metadata["backbone"], plan["seed"], pretrained=False)
    model.load_state_dict(torch.load(args.cache/"initial_model.pt", map_location="cpu", weights_only=True))
    model.to(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    facts = dict(protocol="alternative-whole-patient-v2", complete=False, valid_for_results=False,
                 settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 plan_sha256=sha(args.plan), cache_sha256=sha(args.cache/"cache.json"),
                 source_sha256=source_hashes(Path(plan["repo"])), runtime=runtime(torch),
                 git_commit=git_head(plan["repo"]),
                 patient_update="one global attention prediction, one BCE, one optimizer step per patient per epoch",
                 evaluation_scope="development; validation selects checkpoints, not an unbiased outer-fold test",
                 unbiased_outer_estimate=False, smoke=args.smoke,
                 validation="same complete real-only development pools, global attention", best_cycles={}, stages={})
    write_json(args.out/"run_record.json", facts)
    pixels = Pixels(plan, normalizer_from(features.metadata["normalization"]), args.out)
    pools = plan["pools"][args.arm]
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(plan["class_weight"], device=args.device))
    overall_best, selected_stage = float("inf"), None
    try:
        for stage, cycles, rate in (("frozen", args.frozen_cycles, args.frozen_lr),
                                     ("finetune", args.finetune_cycles, args.finetune_lr)):
            frozen = stage == "frozen"
            if frozen:
                model.backbone.requires_grad_(False)
                optimizer = torch.optim.Adam(model.pool.parameters(), lr=rate)
            else:
                encoder_parameters = finetune_parameters(model, features.metadata["backbone"])
                optimizer = torch.optim.Adam([
                    {"params": encoder_parameters, "lr": args.encoder_lr},
                    {"params": model.pool.parameters(), "lr": rate}])
                facts["trainable_encoder_parameters"] = [name for name, p in model.backbone.named_parameters() if p.requires_grad]
            monitor, best_cycle = Plateau(args.patience, args.min_delta), 0
            reason = "stage epoch limit"
            for cycle in range(cycles):
                started, train_loss = time.monotonic(), 0.0
                model.train()
                if frozen:
                    model.backbone.eval()
                patients = sorted(pools)
                rng(plan["seed"], stage, cycle, "patients").shuffle(patients)
                for patient in patients:
                    tiles = pools[patient]
                    target = torch.tensor([float(tiles[0]["label"])], device=args.device)
                    optimizer.zero_grad(set_to_none=True)
                    if frozen:
                        embeddings = features.get(patient, tiles).unsqueeze(0).to(args.device)
                        logits, _ = model.pool(embeddings)
                        loss = loss_fn(logits, target)
                        loss.backward()
                        train_loss += loss.item()
                    else:
                        chunks = coverage_bags(tiles, args.bag_size, plan["seed"], cycle)
                        def load_chunk(index):
                            tensors = []
                            for row in chunks[index]:
                                seed = rng(plan["seed"], cycle, row["tile_id"], "augment").getrandbits(63)
                                tensors.append(pixels.tile(row, seed))
                            return torch.stack(tensors)
                        result = backward_patient(model, chunks, load_chunk, target, loss_fn, args.device)
                        train_loss += result["loss"]
                        for chunk_index, chunk in enumerate(chunks):
                            append(args.out/"chunk_audit.csv", dict(cycle=cycle+1, patient_id=patient,
                                   chunk=chunk_index, real=sum(r["kind"] == "real" for r in chunk),
                                   synthetic=sum(r["kind"] == "synthetic" for r in chunk),
                                   tile_ids=json.dumps([r["tile_id"] for r in chunk])))
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError("Nonfinite gradients; stopped before optimizer update")
                    optimizer.step()
                validation_loss, auc, predictions = predict(model, plan, features, pixels, args, frozen)
                statistics = summarize(predictions)
                append(args.out/"metrics.csv", dict(stage=stage, cycle=cycle+1,
                       training_loss=train_loss/len(patients), validation_loss=validation_loss,
                       validation_auroc=auc, accuracy=statistics["accuracy"],
                       balanced_accuracy=statistics["balanced_accuracy"],
                       tn=statistics["tn"], fp=statistics["fp"], fn=statistics["fn"], tp=statistics["tp"],
                       seconds=time.monotonic()-started))
                for row in predictions:
                    append(args.out/"validation_predictions.csv", dict(stage=stage, cycle=cycle+1, **row))
                print(f"{stage} cycle {cycle+1}: validation loss {validation_loss:.5f}, AUROC {auc}", flush=True)
                improved, stop = monitor.observe(validation_loss)
                if improved:
                    best_cycle = cycle+1
                    save_model(args.out/f"{stage}_best.pt", model)
                if stop:
                    reason = "development loss plateau"
                    break
            if not best_cycle:
                raise ValueError("No finite validation loss for checkpoint selection")
            model.load_state_dict(torch.load(args.out/f"{stage}_best.pt", map_location=args.device, weights_only=True))
            facts["best_cycles"][stage] = best_cycle
            facts["stages"][stage] = dict(epochs_run=cycle+1, stop_reason=reason, best_loss=monitor.best)
            if monitor.best < overall_best:
                overall_best, selected_stage = monitor.best, stage
                save_model(args.out/"selected_best.pt", model)
            write_json(args.out/"run_record.json", facts)
        model.load_state_dict(torch.load(args.out/"selected_best.pt", map_location=args.device, weights_only=True))
        _, auc, predictions = predict(model, plan, features, pixels, args, frozen=selected_stage == "frozen")
        statistics = summarize(predictions)
        statistics.update(auroc=auc, evaluation_scope="development", smoke=args.smoke)
        training_labels = [tiles[0]["label"] for tiles in pools.values()]
        majority = int(sum(training_labels) > len(training_labels)/2)
        statistics.update(training_majority_label=majority,
                          training_majority_predictor_accuracy=sum(r["label"] == majority for r in predictions)/len(predictions))
        write_json(args.out/"patient_metrics.json", statistics)
        with (args.out/"confusion_matrix.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_class", "predicted_unmethylated", "predicted_methylated"])
            for label, row in zip(("unmethylated", "methylated"), statistics["confusion_matrix"]):
                writer.writerow([label, *row])
        for row in predictions:
            append(args.out/"predictions.csv", row)
        facts.update(complete=True, valid_for_results=not args.smoke, validation_auroc=auc,
                     selected_stage=selected_stage, epochs_run=sum(s["epochs_run"] for s in facts["stages"].values()),
                     checkpoint_sha256=sha(args.out/"selected_best.pt"))
        write_json(args.out/"run_record.json", facts)
    finally:
        pixels.close()
