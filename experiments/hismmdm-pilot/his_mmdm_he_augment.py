#!/usr/bin/env python3
"""Stage reviewed TCGA references and run the frozen HMU-1st H&E pilot."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

COMMIT = "26a1f9f12a8a3d1799496fb7dacebd722d92f3e3"
ARCHIVE_MD5 = "7354dfba1481afea9af0672476a04ca8"
WEIGHT_NAME = "diffusion_model308000.pt"
WEIGHT_MEMBER = ("pretrained_checkpoints", "IHC_virtual_staining", WEIGHT_NAME)
WEIGHT_SHA256 = "a2dd28c95984770e30ef30e90b9ea95d5bbc3f3ed5313dc44b936335233defc3"
HE = 10
SIZE = 128
QC_FIELDS = ("tissue_fraction", "laplacian_variance", "near_white_fraction", "near_black_fraction")
REF_FIELDS = ("patch_id", "morphology", "qc_defect_flags", *QC_FIELDS)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm="sha256"):
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        table = csv.DictReader(handle)
        if not set(required).issubset(table.fieldnames or ()):
            raise ValueError(f"{path} needs columns {', '.join(required)}.")
        rows = list(table)
    if not rows:
        raise ValueError(f"No rows in {path}.")
    return rows


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_hash(path, expected):
    if not isinstance(expected, str) or digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}.")


def require_new(path):
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def unpack_weights(args):
    require_new(args.output)
    if digest(args.archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Archive does not match the publisher's checkpoint bundle MD5.")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staging = Path(tmp)
        checkpoint = staging / WEIGHT_NAME
        with tarfile.open(args.archive, "r|gz") as bundle:
            for member in bundle:
                parts = Path(member.name).parts
                if parts[-len(WEIGHT_MEMBER):] == WEIGHT_MEMBER and member.isfile():
                    with bundle.extractfile(member) as source, checkpoint.open("xb") as target:
                        shutil.copyfileobj(source, target)
                    break
        if not checkpoint.is_file():
            raise ValueError("The verified bundle did not contain the HMU-1st diffusion weights.")
        require_hash(checkpoint, WEIGHT_SHA256)
        save_json(staging / "weights.json", {
            "family": "HMU-1st", "archive_md5": ARCHIVE_MD5, "archive_member": member.name,
            "filename": WEIGHT_NAME, "sha256": WEIGHT_SHA256, "prepared_at": stamp(),
        })
        staging.rename(args.output)
    print(f"Verified checkpoint archive and staged {args.output / WEIGHT_NAME}.")


def unique_by(rows, field):
    keys = [row[field] for row in rows]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError(f"Empty or duplicate {field} values.")
    return dict(zip(keys, rows))


def training_ids(folds, fold):
    rows = read_csv(folds, ("source", "patient_id", "fold"))
    for row in rows:
        source, patient, assigned = (row[key] for key in ("source", "patient_id", "fold"))
        if source == "TCGA" and (not re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", patient)
                                 or assigned not in {"0", "1", "2", "3", "4"}):
            raise ValueError(f"Invalid TCGA split row for {patient}.")
        if source not in {"TCGA", "CPTAC"} or (source == "CPTAC" and assigned != "external"):
            raise ValueError(f"Invalid cohort/fold mapping for {patient}.")
    return {patient for patient, row in unique_by(rows, "patient_id").items()
            if row["source"] == "TCGA" and row["fold"] != str(fold)}


def check_qc(row):
    for field in QC_FIELDS:
        value = float(row[field])
        if not math.isfinite(value) or value < 0 or (field != "laplacian_variance" and value > 1):
            raise ValueError(f"Invalid {field} for {row['patch_id']}.")
    if not row["morphology"].strip() or row["qc_defect_flags"].strip().lower() != "none":
        raise ValueError(f"{row['patch_id']} must have a reviewed morphology and qc_defect_flags=none.")


def prepare_references(args):
    from PIL import Image
    sys.path.insert(0, str(args.team.resolve() / "data_prep"))
    import common
    import loader

    require_new(args.output)
    folds = args.team / "manifests" / "patient_folds.csv"
    tiling = args.team / "configs" / "tiling.yaml"
    selected = unique_by(read_csv(args.select, REF_FIELDS), "patch_id")
    if len(selected) > 12:
        raise ValueError("The one-day pilot supports at most 12 references per run.")
    labels = unique_by(read_csv(args.labels, ("patient_id", "mgmt_status")), "patient_id")
    allowed = training_ids(folds, args.fold)
    tiling_config, tiling_hash, _ = common.load_tiling_config(tiling)
    if (tiling_config["tile_px"], tiling_config["target_mpp"]) != (224, 0.5):
        raise ValueError("Expected the study's 224 px / 0.5 µm/px tiling config.")
    settings_path = args.modeling / {
        "export": "export_settings.json", "package": loader.PACKAGE_MANIFEST,
        "combined": loader.COMBINED_SETTINGS,
    }[loader.folder_kind(args.modeling)]
    settings = read_json(settings_path)
    if (settings["folds_sha256"] != digest(folds) or settings["config_sha256"] != tiling_hash
            or settings["validation_fold"] != args.fold):
        raise ValueError("QC export fold or configuration differs from the supplied team checkout.")
    tracked = {"folds": folds, "tiling": tiling, "settings": settings_path,
               "modeling": args.modeling / "modeling_patches.csv",
               "selection": args.select, "labels": args.labels}
    hashes = {name: digest(path) for name, path in tracked.items()}
    tiles = loader.with_roles(loader.load_modeling(args.modeling, allow_partial=args.allow_partial), args.fold)
    tiles = tiles[tiles["patch_id"].isin(selected)]
    if tiles["patch_id"].duplicated().any() or set(tiles["patch_id"]) != selected.keys():
        raise ValueError("Each selected reference must appear exactly once in the checked modeling export.")
    eligible = []
    for row in tiles.to_dict("records"):
        patient = row["patient_id"]
        patch = row["patch_id"]
        if row["source"] != "TCGA" or row["role"] != "train" or patient not in allowed:
            raise ValueError(f"Excluded reference: {patch} ({patient}).")
        qc_fields = {field: selected[patch][field] for field in REF_FIELDS if field != "patch_id"}
        check_qc({"patch_id": patch, **qc_fields})
        label = labels[patient]["mgmt_status"].strip().lower()
        if label not in {"methylated", "unmethylated"}:
            raise ValueError(f"Missing verified MGMT class for {patient}.")
        eligible.append((row, label, qc_fields))

    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staging = Path(tmp)
        (staging / "originals").mkdir()
        shutil.copyfile(folds, staging / "patient_folds.csv")
        refs = []
        with loader.TileReader(args.modeling, tiling_config=tiling) as reader:
            for i, (row, label, qc_fields) in enumerate(eligible, 1):
                ref = f"ref{i:02d}"
                rgb = reader.read(row)
                if rgb.shape != (224, 224, 3) or rgb.dtype.name != "uint8":
                    raise ValueError(f"Unexpected RGB image for {row['patch_id']}.")
                rel = f"originals/{ref}.png"
                Image.fromarray(rgb).save(staging / rel)
                refs.append({
                    "reference_id": ref, "source": "TCGA", "patient_id": row["patient_id"],
                    "slide_id": row["slide_id"], "patch_id": row["patch_id"], "assigned_fold": row["fold"],
                    "source_path": row["slide_relpath"], "x0": row["x0"], "y0": row["y0"],
                    "x1": row["x1"], "y1": row["y1"], "mgmt_patient_label": label,
                    "source_mpp": 0.5, "qc_decided_by": row["decided_by"],
                    "image_path": rel, "source_sha256": digest(staging / rel),
                    "pixel_sha256": row["pixel_sha256"], **qc_fields,
                })
        save_csv(staging / "input_manifest.csv", refs)
        for name, path in tracked.items():
            require_hash(path, hashes[name])
        save_json(staging / "references.json", {
            "held_out_fold": args.fold, "manifest_sha256": digest(staging / "input_manifest.csv"),
            "inputs_sha256": hashes, "modeling_folder": str(args.modeling.resolve()),
            "prepared_at": stamp(),
        })
        staging.rename(args.output)
    print(f"Staged {len(refs)} eligible references in {args.output}.")


def validated_inputs(args):
    from PIL import Image

    refs = args.refs
    record = read_json(refs / "references.json")
    manifest = refs / "input_manifest.csv"
    require_hash(manifest, record["manifest_sha256"])
    folds = refs / "patient_folds.csv"
    require_hash(folds, record["inputs_sha256"]["folds"])
    if record["held_out_fold"] != args.fold:
        raise ValueError("Prepared reference fold differs from the requested fold.")
    allowed = training_ids(folds, args.fold)
    weights = read_json(args.weights / "weights.json")
    checkpoint = args.weights / WEIGHT_NAME
    if (weights.get("family") != "HMU-1st" or weights.get("archive_md5") != ARCHIVE_MD5
            or weights.get("filename") != WEIGHT_NAME or weights.get("sha256") != WEIGHT_SHA256
            or tuple(Path(weights.get("archive_member", "")).parts[-3:]) != WEIGHT_MEMBER):
        raise ValueError("Checkpoint lacks the verified HMU-1st archive record.")
    require_hash(checkpoint, weights["sha256"])
    rows = read_csv(manifest, ("reference_id", "patient_id", "source", "assigned_fold", "slide_id",
                               "patch_id", "image_path", "source_sha256", "pixel_sha256", "mgmt_patient_label",
                               "source_mpp", *REF_FIELDS[1:]))
    unique_by(rows, "reference_id")
    unique_by(rows, "patch_id")
    assigned = {r["patient_id"]: r["fold"] for r in read_csv(folds, ("patient_id", "fold"))}
    if len(rows) > 12:
        raise ValueError("The one-day pilot supports at most 12 references per run.")
    for row in rows:
        ref = row["reference_id"]
        if (row["source"] != "TCGA" or row["patient_id"] not in allowed
                or row["assigned_fold"] != assigned[row["patient_id"]] or not row["slide_id"]
                or row["mgmt_patient_label"] not in {"methylated", "unmethylated"}):
            raise ValueError(f"Invalid TCGA training lineage for {ref}.")
        check_qc(row)
        rel = f"originals/{ref}.png"
        if not re.fullmatch(r"ref\d{2}", ref) or row["image_path"] != rel or float(row["source_mpp"]) != 0.5:
            raise ValueError(f"Invalid paths or geometry for {ref}.")
        require_hash(refs / rel, row["source_sha256"])
        with Image.open(refs / rel) as im:
            if im.size != (224, 224) or im.mode != "RGB" or im.format != "PNG":
                raise ValueError(f"Expected a lossless 224×224 RGB reference for {ref}.")
            if hashlib.sha256(im.tobytes()).hexdigest() != row["pixel_sha256"]:
                raise ValueError(f"RGB pixels differ from the team export for {ref}.")
    return checkpoint, weights, rows, record, sorted(allowed)


def release_settings(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Use a clean His-MMDM checkout at {COMMIT}.")
    cfg = (repo / "scripts/dataset_specific_configurations/IHC_virtual_staining.sh").read_text()
    mapping = re.search(r"--class_mapping ([^\s]+)", cfg)
    classes = dict(term.split(":") for term in mapping[1].split(",")) if mapping else {}
    if classes.get("HE") != str(HE) or len(classes) != 26 or WEIGHT_NAME not in cfg:
        raise ValueError("Unexpected released HMU-1st class mapping or checkpoint.")
    return {
        "image_size": SIZE, "model_type": "unet", "num_classes": len(classes),
        "attention_resolutions": "32,16,8", "class_cond": True, "dropout": 0.1,
        "learn_sigma": True, "num_channels": 256, "num_heads": 4, "num_res_blocks": 2,
        "resblock_updown": True, "use_new_attention_order": True, "use_scale_shift_norm": True,
        "diffusion_steps": 1000, "noise_schedule": "linear", "timestep_respacing": "", "use_fp16": True,
    }


def make_tensor(image_path, crop, torch, device):
    from PIL import Image
    with Image.open(image_path) as original:
        resized = crop(original, SIZE)
    return torch.from_numpy(resized.copy()).to(device=device, dtype=torch.float32).permute(
        2, 0, 1
    ).unsqueeze(0) / 127.5 - 1


def to_image(value):
    if not value.isfinite().all():
        raise ValueError("DDIM returned nonfinite values.")
    return ((value[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()


def decode(diffusion, model, latent, target, eta, seed, torch, device):
    if target.shape != (1,) or target.item() != HE:
        raise ValueError("Target condition must be HE.")
    with torch.random.fork_rng(devices=[device]):
        torch.cuda.manual_seed(seed)
        return diffusion.ddim_sample_loop(
            model, latent.shape, noise=latent.clone(), model_kwargs={"y": target},
            clip_denoised=True, cond_fn=None, eta=eta, device=device, progress=True,
        )


def blind_montage(path, images):
    from PIL import Image, ImageDraw
    width = SIZE * len(images)
    strip = Image.new("RGB", (width, SIZE + 23), "white")
    draw = ImageDraw.Draw(strip)
    for index, (name, rgb) in enumerate(images):
        strip.paste(Image.fromarray(rgb), (index * SIZE, 23))
        draw.text((index * SIZE + 4, 4), name, fill="black")
    strip.save(path)


def review_passes(args, rows, signature, np):
    required = ("reference_id", "decision", "control_sha256", "reviewer")
    reviews = unique_by(read_csv(args.review, required), "reference_id")
    if set(reviews) != {row["reference_id"] for row in rows}:
        raise ValueError("Review sheet must contain an explicit pass or fail for each reconstruction.")
    previous = read_json(args.reconstruction / "completed.json")
    if previous["signature"] != signature or previous["phase"] != "reconstruct":
        raise ValueError("Reconstruction receipt does not match this run's inputs, weights or code.")
    passed = {}
    for ref, review in reviews.items():
        if review["decision"] not in {"pass", "fail"} or not review["reviewer"].strip():
            raise ValueError(f"Enter a pass/fail and reviewer for {ref}.")
        control = previous["controls"][ref]
        require_hash(args.reconstruction / f"reconstruction_eta0/{ref}.png", control["sha256"])
        if review["control_sha256"] != control["sha256"]:
            raise ValueError(f"Review for {ref} refers to a different reconstruction.")
        if review["decision"] == "pass":
            latent_file = args.reconstruction / f"latents/{ref}.npy"
            require_hash(latent_file, control["latent_sha256"])
            latent = np.load(latent_file, allow_pickle=False)
            if latent.shape != (1, 3, SIZE, SIZE) or latent.dtype != np.float32 or not np.isfinite(latent).all():
                raise ValueError(f"Invalid encoded reference for {ref}.")
            passed[ref] = latent
    if not passed:
        raise ValueError("No reference passed the reconstruction review.")
    return passed, previous["controls"], previous["runtime"]


def run_pilot(args):
    repo = args.upstream.resolve()
    config = release_settings(repo)
    checkpoint, weights, references, preparation, allowlist = validated_inputs(args)
    require_new(args.output)
    if args.phase == "reconstruct" and (args.review or args.reconstruction):
        raise ValueError("Use reconstruction and review paths only with the variants phase.")
    if args.phase == "variants" and not (args.review and args.reconstruction):
        raise ValueError("Variants require a completed reconstruction directory and its review sheet.")
    signature = {
        "checkpoint_sha256": weights["sha256"], "resolved_model_flags": config,
        "checkpoint_member": weights["archive_member"], "repository_commit": COMMIT,
        "wrapper_sha256": digest(Path(__file__).resolve()), "fold": args.fold,
        "manifest_sha256": preparation["manifest_sha256"],
        "folds_sha256": preparation["inputs_sha256"]["folds"],
        "source_domain": "HE", "source_class_id": HE, "target_domain": "HE", "target_class_id": HE,
        "encoding": {"sampler": "DDIM inversion", "eta": 0.0, "clip_denoised": False},
        "decoding": {"sampler": "DDIM", "clip_denoised": True, "classifier_guidance": False},
        "latent_edits": False, "frozen": True, "normalization": "RGB / 127.5 - 1",
        "native_model_input_px": SIZE, "source_mpp": 0.5, "source_px": 224,
        "source_fov_um": 112.0, "effective_input_mpp": 0.875,
        "model_input_preprocessing": "upstream center_crop_arr (bicubic resize for square 224 px input)",
    }
    preflight = {
        "signature": signature, "phase": args.phase, "decoding_eta": args.etas if args.phase == "variants" else [0.0],
        "seeds": args.seeds, "reference_count": len(references), "allowed_training_patient_ids": allowlist,
        "device": args.device, "dry_run": args.dry_run, "time_utc": stamp(),
    }
    args.output.mkdir()
    save_json(args.output / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)

    import numpy as np
    if args.phase == "variants":
        passed, controls, expected_runtime = review_passes(args, references, signature, np)
        preflight["passed_reference_ids"] = sorted(passed)
        preflight["review_sha256"] = digest(args.review)
        save_json(args.output / "preflight.json", preflight)
    if args.dry_run:
        print("Preflight complete. See preflight.json for the resolved settings.")
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity
    sys.path.insert(0, str(repo))
    from cg_diffusion.image_datasets import center_crop_arr
    from cg_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    if not torch.cuda.is_available():
        raise RuntimeError("Run inference in the GPU-enabled Run:ai environment.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "pillow": Image.__version__, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device), "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    if args.phase == "variants" and runtime != expected_runtime:
        raise ValueError("Use the same GPU type and software environment as the reconstruction controls.")
    save_json(args.output / "runtime.json", runtime)
    model, diffusion = create_model_and_diffusion(**(model_and_diffusion_defaults() | config))
    if diffusion.num_timesteps != 1000 or list(diffusion.timestep_map) != list(range(1000)):
        raise ValueError("Unexpected effective DDIM step schedule.")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True), strict=True)
    model.requires_grad_(False).eval().to(device)
    model.convert_to_fp16()

    def model_fn(x, t, y):
        if y.shape != (1,) or y.item() != HE or torch.is_grad_enabled():
            raise RuntimeError("Both model paths must use frozen, gradient-free H&E inference.")
        return model(x, t, y=y)

    for folder in ("native_inputs", "montages", "latents" if args.phase == "reconstruct" else "variants",
                   "reconstruction_eta0" if args.phase == "reconstruct" else "technical_metrics"):
        (args.output / folder).mkdir()
    results = []
    if args.phase == "reconstruct":
        controls = {}
        review_template = []
    with (args.output / "run_log.jsonl").open("a", encoding="utf-8") as log, torch.inference_mode():
        for row in references:
            ref = row["reference_id"]
            if args.phase == "variants" and ref not in passed:
                continue
            try:
                source = torch.tensor([HE], dtype=torch.long, device=device)
                target = torch.tensor([HE], dtype=torch.long, device=device)
                input_tensor = make_tensor(args.refs / row["image_path"], center_crop_arr, torch, device)
                native = (input_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).round().astype(np.uint8)
                Image.fromarray(native).save(args.output / f"native_inputs/{ref}.png")
                shown = [("Native input", native)]
                metric = lambda a, b: float(structural_similarity(a, b, channel_axis=2, data_range=255))
                if args.phase == "reconstruct":
                    if source.item() != HE:
                        raise ValueError("Source condition must be HE.")
                    latent = diffusion.ddim_reverse_sample_loop(
                        model_fn, input_tensor, clip_denoised=False, eta=0.0,
                        model_kwargs={"y": source}, cond_fn=None, device=device, progress=True,
                    )
                    if not latent.isfinite().all():
                        raise ValueError("DDIM inversion returned nonfinite values.")
                    latent_file = args.output / f"latents/{ref}.npy"
                    np.save(latent_file, latent.cpu().numpy(), allow_pickle=False)
                    decoded = decode(diffusion, model_fn, latent, target, 0.0, args.seeds[0], torch, device)
                    repeated = decode(diffusion, model_fn, latent, target, 0.0, args.seeds[1], torch, device)
                    if not torch.equal(decoded, repeated):
                        raise ValueError("Deterministic reconstruction changed when only the seed changed.")
                    rgb = to_image(decoded)
                    rel = f"reconstruction_eta0/{ref}.png"
                    Image.fromarray(rgb).save(args.output / rel)
                    sha = digest(args.output / rel)
                    controls[ref] = {"sha256": sha, "latent_sha256": digest(latent_file)}
                    review_template.append({"reference_id": ref, "decision": "", "reviewer": "", "control_sha256": sha})
                    shown.append(("Reconstruction", rgb))
                    generated = [(rel, sha, 0.0, args.seeds[0], metric(native, rgb), 1.0)]
                else:
                    latent = torch.from_numpy(passed[ref]).to(device)
                    control_path = args.reconstruction / f"reconstruction_eta0/{ref}.png"
                    with Image.open(control_path) as im:
                        control = np.array(im)
                    shown.append(("Reconstruction", control))
                    variants = [(eta, seed) for eta in args.etas for seed in args.seeds]
                    random.Random(f"{args.fold}:{ref}:montage").shuffle(variants)
                    generated = []
                    paired_variants = {eta: [] for eta in args.etas}
                    for index, (eta, seed) in enumerate(variants, 1):
                        decoded = decode(diffusion, model_fn, latent, target, eta, seed, torch, device)
                        if seed == args.seeds[0]:
                            repeat = decode(diffusion, model_fn, latent, target, eta, seed, torch, device)
                            if not torch.equal(decoded, repeat):
                                raise ValueError(f"Same-seed stochastic decode did not reproduce at eta {eta}.")
                        shown_id = f"Variant {index:02d}"
                        rgb = to_image(decoded)
                        shown.append((shown_id, rgb))
                        paired_variants[eta].append(rgb)
                        rel = f"variants/{ref}-{index:02d}.png"
                        Image.fromarray(rgb).save(args.output / rel)
                        generated.append((rel, digest(args.output / rel), eta, seed,
                                          metric(native, rgb), metric(control, rgb)))
                    comparisons = []
                    for eta, (a, b) in paired_variants.items():
                        if np.array_equal(a, b):
                            raise ValueError(f"Distinct stochastic seeds produced identical pixels at eta {eta}.")
                        comparisons.append({"eta": eta, "between_seed_ssim": metric(a, b)})
                    save_json(args.output / f"technical_metrics/{ref}.json", comparisons)
                blind_montage(args.output / f"montages/{ref}.png", shown)
                for rel, sha, eta, seed, source_ssim, control_ssim in generated:
                    results.append({
                        **row, "held_out_fold": args.fold, "repo_commit": COMMIT,
                        "checkpoint_sha256": weights["sha256"], "source_domain": "HE", "target_domain": "HE",
                        "eta_encode": 0, "eta_decode": eta, "ddim_steps": 1000, "seed": seed,
                        "variant_id": Path(rel).stem, "output_path": rel, "output_sha256": sha,
                        "model_input_px": SIZE, "ssim_vs_native_input": source_ssim,
                        "ssim_vs_reconstruction": control_ssim, "review_status": "pending",
                    })
                log.write(json.dumps({"reference_id": ref, "status": "completed", "time": stamp()}) + "\n")
            except Exception as error:
                log.write(json.dumps({"reference_id": ref, "status": "failed", "time": stamp(),
                                      "error": f"{type(error).__name__}: {error}"}) + "\n")
                raise
            finally:
                log.flush()

    require_hash(checkpoint, weights["sha256"])
    save_csv(args.output / "generation_manifest.csv", results)
    final = {"phase": args.phase, "signature": signature, "runtime": runtime,
             "outputs": len(results), "completed_at": stamp()}
    if args.phase == "reconstruct":
        save_csv(args.output / "reconstruction_review.csv", review_template)
        final["controls"] = controls
    save_json(args.output / "completed.json", final)
    print(f"Completed {args.phase} with {len(results)} outputs in {args.output}.")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    weights = commands.add_parser("weights", help="Verify the official archive and extract only HMU-1st weights")
    weights.add_argument("--archive", required=True, type=Path)
    weights.add_argument("--output", required=True, type=Path)
    weights.set_defaults(action=unpack_weights)

    refs = commands.add_parser("prepare", help="Read only selected QC-approved, training-fold references")
    for name in ("team", "modeling", "select", "labels", "output"):
        refs.add_argument(f"--{name}", required=True, type=Path)
    refs.add_argument("--fold", required=True, type=int, choices=range(5))
    refs.add_argument("--allow-partial", action="store_true", help="Allow a partial QC export for this feasibility pilot")
    refs.set_defaults(action=prepare_references)

    run = commands.add_parser("run", help="Run preflight, reconstruction or gated stochastic decoding")
    for name in ("upstream", "weights", "refs", "output"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--fold", required=True, type=int, choices=range(5))
    run.add_argument("--phase", choices=("reconstruct", "variants"), required=True)
    run.add_argument("--reconstruction", type=Path)
    run.add_argument("--review", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--etas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    run.add_argument("--seeds", type=int, nargs=2, default=[101, 102])
    run.set_defaults(action=run_pilot)
    args = parser.parse_args()
    if args.action is run_pilot and (
        not re.fullmatch(r"cuda:\d+", args.device) or len(set(args.etas)) != len(args.etas)
        or any(not 0 < eta <= 1 for eta in args.etas) or len(set(args.seeds)) != 2
        or any(not 0 <= seed < 2**63 for seed in args.seeds)
    ):
        parser.error("Use cuda:N, distinct positive eta values <=1 and two distinct nonnegative seeds <2**63.")
    args.action(args)


if __name__ == "__main__":
    try:
        cli()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Pilot stopped: {error}")
