#!/usr/bin/env python3
"""Matched four-corner His-MMDM arm; prepare on CPU, benchmark/run on a full GPU."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import his_mmdm_he_augment as core

PROTOCOL = "frozen-hmu1st-he-corners-v1"
CORNERS = {"tl": (0, 0), "tr": (96, 0), "bl": (0, 96), "br": (96, 96)}
LABELS = ("methylated", "unmethylated")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def atom(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes its folder: {relative}")
    return path


def label(row):
    binary = {"0": "unmethylated", "1": "methylated"}.get(row.get("mgmt", ""))
    described = row.get("mgmt_status", "").strip().lower()
    require(not described or described in LABELS, "Invalid descriptive MGMT label")
    require(not row.get("mgmt") or binary, "Invalid binary MGMT label")
    require(not (binary and described) or binary == described, "Conflicting MGMT labels")
    require(binary or described, "Missing MGMT label")
    return binary or described


def parent_inputs(parent):
    """Verify existing training lineage without requiring invented per-tile reviews."""
    plan = core.read_json(parent / "batch_plan.json")
    refs = parent / "references"
    prep = core.read_json(refs / "references.json")
    core.require_hash(refs / "references.json", plan["references_sha256"])
    core.require_hash(refs / "input_manifest.csv", prep["manifest_sha256"])
    core.require_hash(parent / "patient_allocation.csv", plan["allocation_sha256"])
    for filename, key in (("patient_folds.csv", "folds"), ("patient_labels.csv", "labels")):
        core.require_hash(refs / filename, prep["inputs_sha256"][key])
    require(plan["protocol"] == "frozen-hmu1st-he-resize-v1"
            and plan["held_out_fold"] in range(5)
            and plan["source_count"] == 2000 and plan["synthetic_count"] == 4000
            and plan["sources_per_class"] == 1000 and plan["shard_count"] == 2
            and plan["variants_per_source"] == 2 and plan["eta"] == 0.3,
            "Expected the existing balanced 2,000-source / 4,000-variant parent")
    rows = core.read_csv(refs / "input_manifest.csv", ("reference_id", "patch_id", "patient_id"))
    require(len(rows) == 2000, "Wrong source count")
    core.unique_by(rows, "reference_id")
    core.unique_by(rows, "patch_id")
    folds = core.unique_by(core.read_csv(refs / "patient_folds.csv", ("patient_id", "fold", "source")), "patient_id")
    labels = {p: label(r) for p, r in core.unique_by(
        core.read_csv(refs / "patient_labels.csv", ("patient_id",)), "patient_id").items()}
    require(prep["held_out_fold"] == plan["held_out_fold"], "Parent fold mismatch")
    for r in rows:
        ref = r["reference_id"]
        require(re.fullmatch(r"ref\d{6}", ref), "Unsafe parent reference ID")
        f = folds[r["patient_id"]]
        require(r["source"] == f["source"] == "TCGA" and r["assigned_fold"] == f["fold"]
                and f["fold"] in {"0", "1", "2", "3", "4"}
                and f["fold"] != str(plan["held_out_fold"]), "Non-training patient in references")
        require(r["mgmt_patient_label"] == labels[r["patient_id"]], "Patient label mismatch")
        require(float(r["source_mpp"]) == 0.5 and r["image_path"] == f"originals/{ref}.png",
                "Wrong parent image geometry/path")
        require(r["repeat_checks"] in {"True", "False"}, "Invalid repeat-check flag")
        require(r.get("qc_defect_flags", "").strip().lower() in {"", "none"}, "Parent QC defect flag")
    for shard in ("0", "1"):
        require(Counter(r["mgmt_patient_label"] for r in rows if r["workload_shard"] == shard)
                == Counter(dict.fromkeys(LABELS, 500)), "Parent shard/class balance mismatch")
    return plan, prep, rows


def crop_rgb(image, corner):
    require(image.mode == "RGB" and image.size == (224, 224), "Expected RGB 224-pixel original")
    x, y = CORNERS[corner]
    return image.crop((x, y, x + 128, y + 128))


def prepare(args):
    from PIL import Image
    parent = args.parent.resolve()
    parent_plan, prep, originals = parent_inputs(parent)
    require(not args.out.exists(), "Output exists; inspect it rather than overwriting")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        stage = Path(temporary) / "batch"
        refs = stage / "references"
        (refs / "originals").mkdir(parents=True)
        (refs / "crops").mkdir()
        for filename in ("patient_folds.csv", "patient_labels.csv"):
            shutil.copyfile(parent / "references" / filename, refs / filename)
        shutil.copyfile(parent / "patient_allocation.csv", stage / "patient_allocation.csv")
        rows = []
        for r in originals:
            source = inside(parent / "references", r["image_path"])
            core.require_hash(source, r["source_sha256"])
            shutil.copyfile(source, refs / r["image_path"])
            with Image.open(source) as im:
                require(im.format == "PNG", "Parent image must be lossless PNG")
                require(hashlib.sha256(im.tobytes()).hexdigest() == r["pixel_sha256"], "Parent pixel mismatch")
                for corner, (x, y) in CORNERS.items():
                    ref = f'{r["reference_id"]}-{corner}'
                    relative = f"crops/{ref}.png"
                    cropped = crop_rgb(im, corner)
                    cropped.save(refs / relative)
                    rows.append({**r, "reference_id": ref, "parent_reference_id": r["reference_id"],
                                 "original_path": r["image_path"], "image_path": relative,
                                 "corner": corner, "crop_x": x, "crop_y": y, "crop_px": 128,
                                 "crop_mpp": 0.5, "crop_fov_um": 64,
                                 "crop_sha256": core.digest(refs / relative),
                                 "crop_pixel_sha256": hashlib.sha256(cropped.tobytes()).hexdigest(),
                                 "crop_qc_status": "not_assessed"})
        core.save_csv(refs / "input_manifest.csv", rows)
        plan = dict(protocol=PROTOCOL, held_out_fold=parent_plan["held_out_fold"], seed=parent_plan["seed"],
                    original_sources=2000, crop_inputs=8000, synthetic_count=16000,
                    variants_per_crop=2, shard_count=2, corners=CORNERS,
                    source_mpp=0.5, crop_px=128, crop_fov_um=64, original_fov_um=112,
                    eta_encode=0.0, eta_decode=0.3, ddim_steps=1000, source_class_id=10, target_class_id=10,
                    parent_batch=str(parent), parent_plan_sha256=core.digest(parent / "batch_plan.json"),
                    parent_references_sha256=core.digest(parent / "references/references.json"),
                    parent_manifest_sha256=prep["manifest_sha256"],
                    manifest_sha256=core.digest(refs / "input_manifest.csv"),
                    folds_sha256=core.digest(refs / "patient_folds.csv"),
                    labels_sha256=core.digest(refs / "patient_labels.csv"),
                    allocation_sha256=core.digest(stage / "patient_allocation.csv"),
                    created_at=core.stamp(), review_status="pending", training_eligible=False)
        atom(stage / "batch_plan.json", plan)
        stage.rename(args.out)
    print(f"Prepared 2,000 originals -> 8,000 corner inputs -> 16,000 planned variants: {args.out}")


def validated(batch):
    plan = core.read_json(batch / "batch_plan.json")
    require(plan["protocol"] == PROTOCOL and plan["original_sources"] == 2000 and plan["crop_inputs"] == 8000
            and plan["synthetic_count"] == 16000 and plan["variants_per_crop"] == 2
            and plan["shard_count"] == 2 and plan["source_mpp"] == 0.5 and plan["crop_px"] == 128
            and plan["crop_fov_um"] == 64 and plan["original_fov_um"] == 112
            and plan["source_class_id"] == 10 and plan["target_class_id"] == 10
            and plan["corners"] == {k: list(v) for k, v in CORNERS.items()}
            and plan["eta_decode"] == 0.3 and plan["eta_encode"] == 0
            and plan["ddim_steps"] == 1000, "Unexpected crop protocol")
    refs = batch / "references"
    for path, key in ((refs / "input_manifest.csv", "manifest_sha256"),
                      (refs / "patient_folds.csv", "folds_sha256"),
                      (refs / "patient_labels.csv", "labels_sha256"),
                      (batch / "patient_allocation.csv", "allocation_sha256")):
        core.require_hash(path, plan[key])
    parent = Path(plan["parent_batch"])
    for path, key in ((parent / "batch_plan.json", "parent_plan_sha256"),
                      (parent / "references/references.json", "parent_references_sha256"),
                      (parent / "references/input_manifest.csv", "parent_manifest_sha256")):
        core.require_hash(path, plan[key])
    _, _, parents = parent_inputs(parent)
    originals = core.unique_by(parents, "reference_id")
    rows = core.read_csv(refs / "input_manifest.csv", ("reference_id", "parent_reference_id", "corner"))
    require(len(rows) == 8000, "Wrong crop input count")
    core.unique_by(rows, "reference_id")
    require(Counter(r["parent_reference_id"] for r in rows) == Counter(dict.fromkeys(originals, 4)),
            "Crops do not cover exactly the original 2,000 sources")
    for r in rows:
        original = originals[r["parent_reference_id"]]
        require(r["corner"] in CORNERS and r["reference_id"] == original["reference_id"] + "-" + r["corner"],
                "Invalid crop identity")
        for field in ("source", "patient_id", "assigned_fold", "slide_id", "patch_id", "mgmt_patient_label",
                      "workload_shard", "repeat_checks", "source_sha256", "pixel_sha256"):
            require(r[field] == original[field], f"Altered parent lineage: {field}")
        require((int(r["crop_x"]), int(r["crop_y"])) == CORNERS[r["corner"]]
                and int(r["crop_px"]) == 128 and float(r["crop_mpp"]) == 0.5,
                "Altered crop geometry")
        require(r["original_path"] == original["image_path"]
                and r["image_path"] == f'crops/{r["reference_id"]}.png', "Invalid image paths")
    return plan, rows


def grouped(rows, shard, batch_size):
    """Fixed membership: repeat-check sources first, then interleave source classes."""
    selected = [r for r in rows if int(r["workload_shard"]) == shard]
    ordered = []
    for repeat in ("True", "False"):
        groups = {c: sorted((r for r in selected if r["repeat_checks"] == repeat
                             and r["mgmt_patient_label"] == c), key=lambda r: r["reference_id"])
                  for c in LABELS}
        # Keep each source's four corners together, alternating patients' class.
        for start in range(0, max(map(len, groups.values()), default=0), 4):
            for c in LABELS:
                ordered.extend(groups[c][start:start + 4])
    require(len(ordered) == len(selected), "Invalid grouping")
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def seed_for(seed, ids, variant):
    return int(sha_object([PROTOCOL, seed, ids, variant])[:15], 16)


def checked_record(folder, group_id, ids, signature):
    path = folder / "records" / f"{group_id}.json"
    if not path.exists():
        return None
    record = core.read_json(path)
    require(record["signature_sha256"] == signature and record["input_ids"] == ids,
            "Resume configuration or batch membership changed")
    expected = {f"{ref}-{v:02d}" for ref in ids for v in (1, 2)}
    require({r["variant_id"] for r in record["outputs"]} == expected
            and len(record["outputs"]) == len(expected), "Incomplete variant receipt")
    for relative, digest in record["file_sha256"].items():
        core.require_hash(inside(folder, relative), digest)
    for row in record["outputs"]:
        for p, h in (("output_path", "output_sha256"), ("classifier_path", "classifier_sha256"),
                     ("reconstruction_path", "reconstruction_sha256")):
            require(record["file_sha256"].get(row[p]) == row[h], "Unprotected output in receipt")
    return record


def worker(args):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    batch = args.batch.resolve()
    plan, rows = validated(batch)
    config = core.release_settings(args.upstream.resolve())
    checkpoint = args.weights / core.WEIGHT_NAME
    weight_record = core.read_json(args.weights / "weights.json")
    require(weight_record.get("family") == "HMU-1st"
            and weight_record.get("archive_md5") == core.ARCHIVE_MD5
            and weight_record.get("sha256") == core.WEIGHT_SHA256
            and tuple(Path(weight_record.get("archive_member", "")).parts[-3:]) == core.WEIGHT_MEMBER,
            "Not the verified HMU-1st checkpoint")
    core.require_hash(checkpoint, core.WEIGHT_SHA256)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    memory = torch.cuda.get_device_properties(device).total_memory / 2**30
    require(memory >= 90, f"Expected this cluster's full 96-GB GPU; only {memory:.2f} GiB visible")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                   numpy=np.__version__, pillow=Image.__version__, cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"], tf32=False)
    all_groups = grouped(rows, args.shard, args.batch_size)
    groups = all_groups[:args.limit_groups] if args.limit_groups else all_groups
    folder = ((args.benchmark_out.resolve() / f"generated/shard-{args.shard}") if args.benchmark_out
              else batch / f"generated/shard-{args.shard}")
    require(bool(args.limit_groups) == bool(args.benchmark_out), "Use limits only with a separate benchmark output")
    require(not args.benchmark_out or not args.benchmark_out.resolve().is_relative_to(batch / "generated"),
            "Benchmark output must be separate from production")
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signature = dict(protocol=PROTOCOL, batch_plan_sha256=core.digest(batch / "batch_plan.json"),
                     wrapper_sha256=core.digest(Path(core.__file__)), runner_sha256=core.digest(Path(__file__)),
                     upstream_commit=core.COMMIT, checkpoint_sha256=core.WEIGHT_SHA256,
                     config=config, runtime=runtime, batch_size=args.batch_size, shard=args.shard,
                     limit_groups=args.limit_groups, rng="one recorded stream per fixed batch and variant",
                     review_status="pending", training_eligible=False)
    signature_hash = sha_object(signature)
    if (folder / "batch_run.json").exists():
        require(core.read_json(folder / "batch_run.json")["signature_sha256"] == signature_hash,
                "Existing output used different code, software, batch size or inputs; do not overwrite")
    else:
        atom(folder / "batch_run.json", dict(signature=signature, signature_sha256=signature_hash,
                                            created_at=core.stamp(), job=os.environ.get("RUNAI_JOB_NAME")))
    records, pending = [], []
    for i, group in enumerate(groups):
        group_id = f"group{i:05d}"
        ids = [r["reference_id"] for r in group]
        done = checked_record(folder, group_id, ids, signature_hash)
        if done:
            records.append(done)
        else:
            pending.append((group_id, group))
    print(f"Shard {args.shard}: {len(records)}/{len(groups)} batches already complete; batch size {args.batch_size}", flush=True)
    if pending:
        sys.path.insert(0, str(args.upstream.resolve()))
        from cg_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        model, diffusion = create_model_and_diffusion(**(model_and_diffusion_defaults() | config))
        require(diffusion.num_timesteps == 1000 and list(diffusion.timestep_map) == list(range(1000)),
                "Changed DDIM timestep schedule")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True), strict=True)
        model.requires_grad_(False).eval().to(device)
        model.convert_to_fp16()

        def model_fn(x, t, y):
            require(y.shape == (len(x),) and bool((y == 10).all()) and not torch.is_grad_enabled(),
                    "Expected frozen H&E -> H&E inference without classifier guidance")
            return model(x, t, y=y)

        def decode(latent, target, eta, seed):
            with torch.random.fork_rng(devices=[device]):
                torch.cuda.manual_seed(seed)
                return diffusion.ddim_sample_loop(model_fn, latent.shape, noise=latent.clone(),
                    model_kwargs={"y": target}, clip_denoised=True, cond_fn=None,
                    eta=eta, device=device, progress=True)

        def rgb(tensor):
            require(bool(tensor.isfinite().all()), "Nonfinite diffusion output")
            return ((tensor.permute(0, 2, 3, 1) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()

        with torch.inference_mode():
            for group_id, group in pending:
                start = time.monotonic()
                ids = [r["reference_id"] for r in group]
                repeat = any(r["repeat_checks"] == "True" for r in group)
                print(f"Starting {group_id}: {len(group)} crops; repeat checks {repeat}", flush=True)
                native = []
                for r in group:
                    path = inside(batch / "references", r["image_path"])
                    core.require_hash(path, r["crop_sha256"])
                    original = inside(batch / "references", r["original_path"])
                    core.require_hash(original, r["source_sha256"])
                    with Image.open(original) as im, Image.open(path) as cropped:
                        expected = crop_rgb(im, r["corner"])
                        require(cropped.mode == "RGB" and cropped.size == (128, 128)
                                and cropped.tobytes() == expected.tobytes(), "Crop pixels changed")
                        require(hashlib.sha256(cropped.tobytes()).hexdigest() == r["crop_pixel_sha256"],
                                "Crop pixel checksum mismatch")
                        native.append(np.array(cropped))
                x = torch.from_numpy(np.stack(native)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1
                target = torch.full((len(group),), 10, device=device, dtype=torch.long)
                torch.cuda.reset_peak_memory_stats(device)
                latent = diffusion.ddim_reverse_sample_loop(model_fn, x, clip_denoised=False, eta=0,
                    model_kwargs={"y": target}, cond_fn=None, device=device, progress=True)
                require(bool(latent.isfinite().all()), "Nonfinite inversion")
                seeds = [seed_for(plan["seed"], ids, v) for v in (1, 2)]
                control_t = decode(latent, target, 0, seeds[0])
                if repeat:
                    require(torch.equal(control_t, decode(latent, target, 0, seeds[1])),
                            "Eta-zero reconstruction depends on RNG seed")
                control = rgb(control_t)
                variants = []
                for v, seed in enumerate(seeds, 1):
                    generated = decode(latent, target, 0.3, seed)
                    if repeat and v == 1:
                        require(torch.equal(generated, decode(latent, target, 0.3, seed)),
                                "Same-seed stochastic decoding did not reproduce")
                    variants.append(rgb(generated))
                require(all(not np.array_equal(variants[0][i], variants[1][i]) for i in range(len(group))),
                        "Different seeds produced identical variant pixels")
                files, outputs = {}, []

                def save(relative, pixels):
                    path = folder / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as f:
                        temporary = Path(f.name)
                    try:
                        Image.fromarray(pixels).save(temporary)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    files[relative] = core.digest(path)
                    return files[relative]

                for i, r in enumerate(group):
                    ref = r["reference_id"]
                    save(f"native_inputs/{ref}.png", native[i])
                    control_path = f"reconstruction_eta0/{ref}.png"
                    control_hash = save(control_path, control[i])
                    for v in (1, 2):
                        pixels = variants[v - 1][i]
                        variant = f"{ref}-{v:02d}"
                        native_path = f"variants/{variant}.png"
                        classifier_path = f"classifier_tiles/{variant}.png"
                        native_hash = save(native_path, pixels)
                        enlarged = np.array(Image.fromarray(pixels).resize((224, 224), Image.Resampling.BICUBIC))
                        classifier_hash = save(classifier_path, enlarged)
                        outputs.append(dict(**r, held_out_fold=plan["held_out_fold"], variant_id=variant,
                            output_path=native_path, output_sha256=native_hash, classifier_path=classifier_path,
                            classifier_sha256=classifier_hash, reconstruction_path=control_path,
                            reconstruction_sha256=control_hash, output_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                            classifier_pixel_sha256=hashlib.sha256(enlarged.tobytes()).hexdigest(),
                            eta_encode=0, eta_decode=0.3, ddim_steps=1000, seed=seeds[v-1],
                            rng_scope="batch", rng_row=i, rng_batch=group_id, model_input_px=128,
                            source_domain="HE", target_domain="HE", classifier_px=224,
                            classifier_effective_mpp=64 / 224,
                            classifier_resize="128 to 224 bicubic; 64 um crop FOV retained",
                            ssim_vs_native_input=float(ssim(native[i], pixels, channel_axis=2, data_range=255)),
                            ssim_vs_reconstruction=float(ssim(control[i], pixels, channel_axis=2, data_range=255)),
                            checkpoint_sha256=core.WEIGHT_SHA256, repo_commit=core.COMMIT,
                            review_status="pending", training_eligible=False))
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - start
                record = dict(input_ids=ids, signature_sha256=signature_hash, outputs=outputs,
                              file_sha256=files, seconds=elapsed, passes=6 if repeat else 4,
                              peak_memory_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
                              completed_at=core.stamp(), job=os.environ.get("RUNAI_JOB_NAME"))
                atom(folder / "records" / f"{group_id}.json", record)
                records.append(record)
                recent = records[-8:]
                rate = sum(r["seconds"] for r in recent) / sum(r["passes"] * len(r["input_ids"]) for r in recent)
                complete_ids = {tuple(r["input_ids"]) for r in records}
                remaining = [g for g in groups if tuple(r["reference_id"] for r in g) not in complete_ids]
                left = sum(len(g) * (6 if any(r["repeat_checks"] == "True" for r in g) else 4) for g in remaining)
                progress = dict(completed_inputs=sum(len(r["input_ids"]) for r in records),
                                expected_inputs=sum(map(len, groups)), completed_outputs=sum(len(r["outputs"]) for r in records),
                                seconds_per_crop_pass=rate, estimated_seconds_remaining=left * rate,
                                updated_at=core.stamp())
                atom(folder / "progress.json", progress)
                print(f'{progress["completed_outputs"]} outputs; {elapsed:.1f} s/batch; ETA {left*rate/3600:.2f} h', flush=True)
                del x, latent, control_t, generated
    checked = [checked_record(folder, f"group{i:05d}", [r["reference_id"] for r in g], signature_hash)
               for i, g in enumerate(groups)]
    require(all(checked), "Missing batch receipts at finalization")
    outputs = [row for record in checked for row in record["outputs"]]
    core.save_csv(folder / "generation_manifest.csv", outputs)
    core.require_hash(checkpoint, core.WEIGHT_SHA256)
    result = dict(protocol=PROTOCOL, phase="benchmark" if args.limit_groups else "generation",
                  outputs=len(outputs), signature_sha256=signature_hash,
                  manifest_sha256=core.digest(folder / "generation_manifest.csv"),
                  completed_at=core.stamp(), review_status="pending", training_eligible=False)
    if args.limit_groups:
        # Exclude the first batch if possible; use identical pass/verification accounting.
        measured = checked[1:] or checked
        rate = sum(r["seconds"] for r in measured) / sum(r["passes"] * len(r["input_ids"]) for r in measured)
        all_passes = sum(len(g) * (6 if any(r["repeat_checks"] == "True" for r in g) else 4)
                         for shard in (0, 1) for g in grouped(rows, shard, args.batch_size))
        result.update(batch_size=args.batch_size, estimated_full_generation_hours=rate * all_passes / 3600,
                      peak_memory_GiB=max(r["peak_memory_GiB"] for r in checked),
                      estimate_excludes="queueing, preemption, model loading, later QC/review and transfer")
    atom(folder / "completed.json", result)
    print(json.dumps(result, indent=2), flush=True)


def inspect(args):
    plan, rows = validated(args.batch)
    total = 0
    for shard in (0, 1):
        folder = args.batch / f"generated/shard-{shard}"
        receipts = list((folder / "records").glob("group*.json"))
        records = [core.read_json(p) for p in receipts]
        count = sum(len(r["outputs"]) for r in records)
        total += count
        print(f"Shard {shard}: {count}/8000 variants; finalized={(folder / 'completed.json').exists()}")
        if args.command == "verify":
            run = core.read_json(folder / "batch_run.json")
            done = core.read_json(folder / "completed.json")
            require(done["phase"] == "generation", "Not a full generation receipt")
            signature = run["signature_sha256"]
            require(sha_object(run["signature"]) == signature == done["signature_sha256"], "Signature mismatch")
            expected = grouped(rows, shard, run["signature"]["batch_size"])
            require(len(receipts) == len(expected), "Unexpected receipt count")
            checked = [checked_record(folder, f"group{i:05d}", [r["reference_id"] for r in g], signature)
                       for i, g in enumerate(expected)]
            require(all(checked), "Missing completed batch")
            expected_rows = [{k: str(v) for k, v in r.items()} for rec in checked for r in rec["outputs"]]
            core.require_hash(folder / "generation_manifest.csv", done["manifest_sha256"])
            actual = core.read_csv(folder / "generation_manifest.csv", ("variant_id",))
            require(actual == expected_rows and len(actual) == done["outputs"] == 8000, "Manifest/receipts disagree")
            require(Counter(r["mgmt_patient_label"] for r in actual) == Counter(dict.fromkeys(LABELS, 4000)),
                    "Output source-class balance changed")
        elif (folder / "progress.json").exists():
            progress = core.read_json(folder / "progress.json")
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(progress["updated_at"])).total_seconds()
            print(f'  Last ETA {progress["estimated_seconds_remaining"]/3600:.2f} h; updated {age:.0f} s ago')
    print(f"Total: {total}/{plan['synthetic_count']}; QC and morphology review remain separate")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--parent", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("worker")
    run.add_argument("--batch", type=Path, required=True)
    run.add_argument("--upstream", type=Path, required=True)
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--shard", type=int, choices=(0, 1), required=True)
    run.add_argument("--batch-size", type=int, choices=(4, 8, 16), required=True)
    run.add_argument("--limit-groups", type=int, default=0)
    run.add_argument("--benchmark-out", type=Path)
    for command in ("status", "verify"):
        s = sub.add_parser(command)
        s.add_argument("--batch", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        require(args.limit_groups >= 0, "Negative batch limit")
        worker(args)
    else:
        inspect(args)


if __name__ == "__main__":
    main()
