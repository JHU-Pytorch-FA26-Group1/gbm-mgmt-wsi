#!/usr/bin/env python3
"""Append variants 03--08 to the ORIGINAL resize sources: 4,000 + 12,000 = 16,000."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import his_mmdm_batch as legacy
import his_mmdm_he_augment as core

PROTOCOL = "original-sources-resize-03-to-08-v1"
VARIANTS = tuple(range(3, 9))
LINEAGE = ("source", "patient_id", "assigned_fold", "slide_id", "patch_id",
           "source_sha256", "pixel_sha256", "mgmt_patient_label")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes its batch")
    return path


def seed_for(plan, ref, index):
    return legacy.seed_for(plan["seed"], ref, "variant", index)


def baseline(parent, deep=True):
    plan, refs = legacy.validate_batch(parent)
    require(plan["protocol"] == "frozen-hmu1st-he-resize-v1", "Wrong original resize protocol")
    require(plan["synthetic_count"] == 2*len(refs), "Original must contain variants 01 and 02 only")
    outputs, by_ref, runs = [], {}, {}
    for shard in (0, 1):
        folder = parent / f"generated/shard-{shard}"
        done, run = core.read_json(folder / "completed.json"), core.read_json(folder / "batch_run.json")
        require(done["signature_sha256"] == run["signature_sha256"], "Original completion signature mismatch")
        core.require_hash(folder / "generation_manifest.csv", done["manifest_sha256"])
        runs[shard] = run
        selected = legacy.source_order([r for r in refs if r["workload_shard"] == str(shard)])
        shard_outputs = []
        for r in selected:
            ref = r["reference_id"]
            record = (legacy.verified_receipt(folder, ref, run["signature_sha256"]) if deep else
                      core.read_json(folder / "records" / f"{ref}.json"))
            require(record is not None and len(record["outputs"]) == 2, "Original source is incomplete")
            expected = {f"{ref}-01", f"{ref}-02"}
            require({v["variant_id"] for v in record["outputs"]} == expected, "Original variant IDs changed")
            for v in record["outputs"]:
                require(all(str(v[k]) == str(r[k]) for k in LINEAGE), "Original variant source lineage differs")
                index = int(v["variant_id"].rsplit("-", 1)[-1])
                require(int(v["seed"]) == seed_for(plan, ref, index), "Original seed mapping differs")
            by_ref[ref] = record
            shard_outputs.extend(record["outputs"])
        require(len(shard_outputs) == done["outputs"] == 2*len(selected), "Wrong original shard count")
        manifest = core.read_csv(folder / "generation_manifest.csv", ("variant_id", "output_sha256"))
        require({(r["variant_id"], r["output_sha256"]) for r in manifest}
                == {(r["variant_id"], r["output_sha256"]) for r in shard_outputs}
                and len(manifest) == len(shard_outputs), "Original manifest and receipts disagree")
        outputs.extend(shard_outputs)
    require(len(outputs) == plan["synthetic_count"], "Original generation count mismatch")
    return plan, refs, by_ref, runs


def matched_sources(refs, crop):
    crop_plan = core.read_json(crop / "batch_plan.json")
    require(crop_plan["protocol"] == "frozen-hmu1st-he-corners-v1", "Wrong crop protocol")
    require(crop_plan["original_sources"] == len(refs) and crop_plan["crop_inputs"] == 4*len(refs)
            and crop_plan["synthetic_count"] == 8*len(refs), "Crop budget differs from matched resize target")
    core.require_hash(crop / "references/input_manifest.csv", crop_plan["manifest_sha256"])
    crop_rows = core.read_csv(crop / "references/input_manifest.csv", LINEAGE)
    originals = core.unique_by(refs, "patch_id")
    require(Counter(r["patch_id"] for r in crop_rows) == Counter(dict.fromkeys(originals, 4)),
            "Crop inputs do not cover exactly four views of each original source")
    for r in crop_rows:
        require(all(str(r[k]) == str(originals[r["patch_id"]][k]) for k in LINEAGE),
                "Crop and original resize source lineage differ")
    return crop_plan


def anchors(parent, crop):
    paths = {"parent_plan": parent / "batch_plan.json",
             "parent_references": parent / "references/references.json",
             "parent_manifest": parent / "references/input_manifest.csv",
             "crop_plan": crop / "batch_plan.json",
             "crop_manifest": crop / "references/input_manifest.csv"}
    for shard in (0, 1):
        for name in ("completed.json", "batch_run.json", "generation_manifest.csv"):
            paths[f"parent_shard_{shard}_{name}"] = parent / f"generated/shard-{shard}" / name
    return {k: dict(path=str(p.resolve()), sha256=core.digest(p)) for k, p in paths.items()}


def prepare(a):
    parent, crop = a.parent.resolve(), a.crop.resolve()
    require(not a.out.exists(), "Output exists; reuse it with validate/submit instead of preparing again")
    original, refs, old_records, _ = baseline(parent)
    crop_plan = matched_sources(refs, crop)
    require(crop_plan["held_out_fold"] == original["held_out_fold"], "Held-out fold differs between designs")
    require(crop_plan["parent_plan_sha256"] == core.digest(parent / "batch_plan.json"),
            "Crop run was prepared from a different original plan")
    old_seeds = {int(v["seed"]) for rec in old_records.values() for v in rec["outputs"]}
    new_seeds = [seed_for(original, r["reference_id"], i) for r in refs for i in VARIANTS]
    require(len(set(new_seeds)) == len(new_seeds) and not (set(new_seeds) & old_seeds),
            "New seeds collide with one another or original seeds")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as temporary:
        stage = Path(temporary) / "batch"
        stage.mkdir()
        shutil.copytree(parent / "references", stage / "references")
        shutil.copyfile(parent / "patient_allocation.csv", stage / "patient_allocation.csv")
        plan = dict(protocol=PROTOCOL, parent_batch=str(parent), crop_batch=str(crop),
                    held_out_fold=original["held_out_fold"], seed=original["seed"],
                    source_count=len(refs), existing_outputs=2*len(refs), synthetic_count=6*len(refs),
                    matched_total=8*len(refs), variant_indices=list(VARIANTS),
                    source_selection="NONE: byte-identical copy of the original prepared references",
                    references_sha256=core.digest(stage / "references/references.json"),
                    allocation_sha256=core.digest(stage / "patient_allocation.csv"),
                    anchors=anchors(parent, crop),
                    preparer_sha256=core.digest(Path(__file__)),
                    legacy_sha256=core.digest(Path(legacy.__file__)), core_sha256=core.digest(Path(core.__file__)),
                    created_at=core.stamp(), review_status="pending", training_eligible=False)
        legacy.atomic_json(stage / "batch_plan.json", plan)
        validate(stage)
        stage.rename(a.out)
    print(f"Original source tiles: {len(refs)}; new tile IDs: 0")
    print(f"Existing {plan['existing_outputs']} + additional {plan['synthetic_count']} = {plan['matched_total']} resize variants")
    print("Variants 01/02 retained; new variants 03/04/05/06/07/08; original seed mapping extended")
    print(f"Prepared: {a.out.resolve()}")


def validate(batch):
    plan = core.read_json(batch / "batch_plan.json")
    require(plan["protocol"] == PROTOCOL and plan["variant_indices"] == list(VARIANTS), "Wrong matched extension")
    for r in plan["anchors"].values():
        core.require_hash(Path(r["path"]), r["sha256"])
    core.require_hash(batch / "references/references.json", plan["references_sha256"])
    core.require_hash(batch / "patient_allocation.csv", plan["allocation_sha256"])
    prep = core.read_json(batch / "references/references.json")
    core.require_hash(batch / "references/input_manifest.csv", prep["manifest_sha256"])
    for filename, key in (("patient_folds.csv", "folds"), ("patient_labels.csv", "labels")):
        core.require_hash(batch / "references" / filename, prep["inputs_sha256"][key])
    parent = Path(plan["parent_batch"])
    original, originals = legacy.validate_batch(parent)
    rows = core.read_csv(batch / "references/input_manifest.csv", ("reference_id",))
    require(rows == originals, "Extension source rows differ from originals; source reselection is forbidden")
    require(core.digest(batch / "references/input_manifest.csv") == prep["manifest_sha256"]
            == core.digest(parent / "references/input_manifest.csv"), "Reference manifests differ")
    require(plan["source_count"] == len(rows) and plan["existing_outputs"] == 2*len(rows)
            and plan["synthetic_count"] == 6*len(rows) and plan["matched_total"] == 8*len(rows)
            and plan["seed"] == original["seed"] and plan["held_out_fold"] == original["held_out_fold"],
            "Source count, variant budget, seed or fold changed")
    require(matched_sources(rows, Path(plan["crop_batch"]))["held_out_fold"] == plan["held_out_fold"],
            "Crop and resize held-out folds differ")
    return plan, rows


def checked(folder, row, signature, plan):
    ref = row["reference_id"]
    path = folder / "records" / f"{ref}.json"
    if not path.exists():
        return None
    rec = core.read_json(path)
    require(rec["signature_sha256"] == signature and rec["reference_id"] == ref, "Incompatible resume receipt")
    expected = {f"{ref}-{i:02d}" for i in VARIANTS}
    require(len(rec["outputs"]) == 6 and {v["variant_id"] for v in rec["outputs"]} == expected,
            "Wrong additional variant identities/count")
    for relative, sha in rec["file_sha256"].items():
        core.require_hash(inside(folder, relative), sha)
    for v in rec["outputs"]:
        index = int(v["variant_id"].rsplit("-", 1)[-1])
        require(all(str(v[k]) == str(row[k]) for k in LINEAGE)
                and int(v["seed"]) == seed_for(plan, ref, index), "Output source or seed changed")
        for p, h in (("output_path", "output_sha256"), ("classifier_path", "classifier_sha256"),
                     ("reconstruction_path", "reconstruction_sha256")):
            require(rec["file_sha256"].get(v[p]) == v[h], "Output is not protected by receipt")
    return rec


def worker(a):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    batch = a.batch.resolve()
    plan, rows = validate(batch)
    parent = Path(plan["parent_batch"])
    _, _, old_records, old_runs = baseline(parent)
    a.refs, a.fold, a.phase = batch / "references", plan["held_out_fold"], "batch"
    config = core.release_settings(a.upstream.resolve())
    checkpoint, weights, refs, prep, _ = core.validated_inputs(a)
    require(weights["sha256"] == old_runs[a.shard]["signature"]["checkpoint_sha256"]
            and core.COMMIT == old_runs[a.shard]["signature"]["upstream_commit"],
            "Checkpoint or upstream revision differs from original resize")
    require(config == old_runs[a.shard]["signature"]["model_flags"], "Model flags differ from original resize")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                   numpy=np.__version__, pillow=Image.__version__, cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(device), compute_capability=list(torch.cuda.get_device_capability(device)),
                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    require(runtime == old_runs[a.shard]["signature"]["runtime"],
            "Inference runtime/GPU type differs from original resize; inspect before proceeding")
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader"],
                              text=True, capture_output=True, check=True).stdout.strip()
    folder = batch / f"generated/shard-{a.shard}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = dict(batch_plan_sha256=core.digest(batch / "batch_plan.json"),
                         matched_script_sha256=core.digest(Path(__file__)), core_sha256=core.digest(Path(core.__file__)),
                         legacy_sha256=core.digest(Path(legacy.__file__)), checkpoint_sha256=weights["sha256"],
                         upstream_commit=core.COMMIT, model_flags=config, runtime=runtime, shard=a.shard,
                         batch_size=1, eta_encode=0, eta_decode=.3, steps=1000, class_id=10,
                         variant_indices=list(VARIANTS), native_preprocessing="original full-field upstream center_crop_arr")
        sig = digest_object(signature)
        config_path = folder / "batch_run.json"
        if config_path.exists():
            require(core.read_json(config_path)["signature_sha256"] == sig, "Resume inputs/code/runtime changed")
        else:
            legacy.atomic_json(config_path, dict(signature=signature, signature_sha256=sig,
                                                created_at=core.stamp(), hardware=hardware))
        selected = legacy.source_order([r for r in refs if r["workload_shard"] == str(a.shard)])
        records, pending = [], []
        for row in selected:
            rec = checked(folder, row, sig, plan)
            (records if rec else pending).append(rec if rec else row)
        print(f"Shard {a.shard}: {len(records)}/{len(selected)} original sources complete; six new variants each", flush=True)
        if pending:
            sys.path.insert(0, str(a.upstream.resolve()))
            from cg_diffusion.image_datasets import center_crop_arr
            from cg_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
            model, diffusion = create_model_and_diffusion(**(model_and_diffusion_defaults() | config))
            require(diffusion.num_timesteps == 1000 and list(diffusion.timestep_map) == list(range(1000)), "DDIM schedule differs")
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True), strict=True)
            model.requires_grad_(False).eval().to(device)
            model.convert_to_fp16()

            def model_fn(x, t, y):
                require(y.shape == (1,) and y.item() == 10 and not torch.is_grad_enabled(), "Expected frozen H&E inference")
                return model(x, t, y=y)

            target = torch.tensor([10], dtype=torch.long, device=device)
            with torch.inference_mode():
                for row in pending:
                    start = time.monotonic()
                    ref, repeat = row["reference_id"], row["repeat_checks"] == "True"
                    print(f"Starting {ref}; variants 03--08; repeat/baseline checks {repeat}", flush=True)
                    old_folder = parent / f"generated/shard-{a.shard}"
                    old = old_records[ref]
                    old_pixels = {v["output_pixel_sha256"] for v in old["outputs"]}
                    old_first = next(v for v in old["outputs"] if v["variant_id"] == f"{ref}-01")
                    x = core.make_tensor(a.refs / row["image_path"], center_crop_arr, torch, device)
                    native = (x[0].permute(1, 2, 0).cpu().numpy()*127.5 + 127.5).round().astype(np.uint8)
                    with Image.open(old_folder / f"native_inputs/{ref}.png") as im:
                        require(np.array_equal(native, np.asarray(im)), "Native input differs from original resize")
                    files = {}

                    def save(relative, pixels):
                        path = folder / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(pixels).save(path)
                        files[relative] = core.digest(path)
                        return files[relative]

                    save(f"native_inputs/{ref}.png", native)
                    latent = diffusion.ddim_reverse_sample_loop(model_fn, x, clip_denoised=False, eta=0,
                        model_kwargs={"y": target}, cond_fn=None, device=device, progress=True)
                    require(bool(latent.isfinite().all()), "Nonfinite inversion")
                    seeds = [seed_for(plan, ref, i) for i in VARIANTS]
                    control_t = core.decode(diffusion, model_fn, latent, target, 0, seeds[0], torch, device)
                    control = core.to_image(control_t)
                    with Image.open(inside(old_folder, old_first["reconstruction_path"])) as im:
                        require(np.array_equal(control, np.asarray(im)), "Reconstruction differs from original resize")
                    if repeat:
                        require(torch.equal(control_t, core.decode(diffusion, model_fn, latent, target, 0, seeds[1], torch, device)),
                                "Eta-zero reconstruction depends on seed")
                        replay = core.decode(diffusion, model_fn, latent, target, .3, int(old_first["seed"]), torch, device)
                        require(hashlib.sha256(core.to_image(replay).tobytes()).hexdigest() == old_first["output_pixel_sha256"],
                                "Original variant-01 replay differs; stop before adding variants")
                        del replay
                    control_path = f"reconstruction_eta0/{ref}.png"
                    control_hash = save(control_path, control)
                    outputs, seen = [], set(old_pixels)
                    for index, seed in zip(VARIANTS, seeds):
                        decoded = core.decode(diffusion, model_fn, latent, target, .3, seed, torch, device)
                        if repeat and index == 3:
                            require(torch.equal(decoded, core.decode(diffusion, model_fn, latent, target, .3, seed, torch, device)),
                                    "Same-seed generation failed to reproduce")
                        pixels = core.to_image(decoded)
                        pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
                        require(pixel_hash not in seen, "Duplicate original/new variant pixels")
                        seen.add(pixel_hash)
                        variant = f"{ref}-{index:02d}"
                        output_path, classifier_path = f"variants/{variant}.png", f"classifier_tiles/{variant}.png"
                        out_hash = save(output_path, pixels)
                        enlarged = np.asarray(Image.fromarray(pixels).resize((224, 224), Image.Resampling.BICUBIC))
                        cls_hash = save(classifier_path, enlarged)
                        outputs.append(dict(**row, variant_id=variant, variant_index=index, held_out_fold=a.fold,
                            output_path=output_path, output_sha256=out_hash, output_pixel_sha256=pixel_hash,
                            classifier_path=classifier_path, classifier_sha256=cls_hash,
                            classifier_pixel_sha256=hashlib.sha256(enlarged.tobytes()).hexdigest(),
                            reconstruction_path=control_path, reconstruction_sha256=control_hash,
                            seed=seed, eta_encode=0, eta_decode=.3, ddim_steps=1000,
                            source_domain="HE", target_domain="HE", model_input_px=128, classifier_px=224,
                            classifier_resize="128 to 224 bicubic", source_fov_um=112,
                            ssim_vs_native_input=float(ssim(native, pixels, channel_axis=2, data_range=255)),
                            ssim_vs_reconstruction=float(ssim(control, pixels, channel_axis=2, data_range=255)),
                            checkpoint_sha256=weights["sha256"], repo_commit=core.COMMIT,
                            review_status="pending", training_eligible=False))
                    record = dict(reference_id=ref, signature_sha256=sig, outputs=outputs, file_sha256=files,
                                  seconds=time.monotonic()-start, passes=11 if repeat else 8,
                                  completed_at=core.stamp(), hardware=hardware)
                    legacy.atomic_json(folder / "records" / f"{ref}.json", record)
                    records.append(record)
                    recent = records[-12:]
                    rate = sum(r["seconds"] for r in recent) / sum(r["passes"] for r in recent)
                    complete = {r["reference_id"] for r in records}
                    left = sum(11 if r["repeat_checks"] == "True" else 8 for r in selected if r["reference_id"] not in complete)
                    legacy.atomic_json(folder / "progress.json", dict(completed_sources=len(records),
                        completed_variants=6*len(records), total_variants=6*len(selected), seconds_per_pass=rate,
                        estimated_seconds_remaining=rate*left, updated_at=core.stamp()))
                    print(f"Saved {6*len(records)}/{6*len(selected)} new variants; shard ETA {timedelta(seconds=round(left*rate))}", flush=True)
                    del x, latent, control_t, decoded
        final = [checked(folder, r, sig, plan) for r in selected]
        require(all(final), "Missing source receipts")
        outputs = [v for rec in final for v in rec["outputs"]]
        core.save_csv(folder / "generation_manifest.csv", outputs)
        core.require_hash(checkpoint, weights["sha256"])
        legacy.atomic_json(folder / "completed.json", dict(outputs=len(outputs), signature_sha256=sig,
            manifest_sha256=core.digest(folder / "generation_manifest.csv"), completed_at=core.stamp(),
            review_status="pending", training_eligible=False))
        print(f"Additional shard {a.shard} complete: {len(outputs)} variants", flush=True)


def verify(batch):
    plan, refs = validate(batch)
    parent = Path(plan["parent_batch"])
    _, _, old, _ = baseline(parent)
    combined = []
    for shard in (0, 1):
        folder = batch / f"generated/shard-{shard}"
        done, run = core.read_json(folder / "completed.json"), core.read_json(folder / "batch_run.json")
        require(done["signature_sha256"] == run["signature_sha256"], "Additional completion signature mismatch")
        core.require_hash(folder / "generation_manifest.csv", done["manifest_sha256"])
        selected = [r for r in refs if r["workload_shard"] == str(shard)]
        additional = [checked(folder, r, run["signature_sha256"], plan) for r in selected]
        require(all(additional) and done["outputs"] == 6*len(selected), "Incomplete additional shard")
        manifest = core.read_csv(folder / "generation_manifest.csv", ("variant_id", "output_sha256"))
        values = [v for rec in additional for v in rec["outputs"]]
        require(len(manifest) == len(values) and {(v["variant_id"], v["output_sha256"]) for v in manifest}
                == {(v["variant_id"], v["output_sha256"]) for v in values}, "Additional manifest and receipts disagree")
        for row, rec in zip(selected, additional):
            values = old[row["reference_id"]]["outputs"] + rec["outputs"]
            require({v["variant_id"] for v in values} == {f'{row["reference_id"]}-{i:02d}' for i in range(1, 9)},
                    "Original source does not have exactly variants 01--08")
            require(len({v["output_pixel_sha256"] for v in values}) == 8, "Duplicate pixels within a source")
            for v in values:
                root = parent if int(v["variant_id"].rsplit("-", 1)[-1]) <= 2 else batch
                combined.append(dict(v, artifact_batch=str(root.resolve()), artifact_batch_name=root.name,
                                     artifact_shard=f"generated/shard-{shard}"))
    require(len(combined) == plan["matched_total"], "Wrong combined resize budget")
    require(Counter(v["mgmt_patient_label"] for v in combined)
            == Counter(dict.fromkeys(legacy.LABELS, 4*len(refs))), "Combined source-class balance changed")
    fields = list(dict.fromkeys(k for r in combined for k in r))
    core.save_csv(batch / "matched_resize_16000.csv", [{k: r.get(k, "") for k in fields} for r in combined])
    legacy.atomic_json(batch / "completed.json", dict(existing_outputs=plan["existing_outputs"],
        additional_outputs=plan["synthetic_count"], matched_total=len(combined), verified_at=core.stamp(),
        manifest_sha256=core.digest(batch / "matched_resize_16000.csv"), review_status="pending", training_eligible=False))
    print(f"Verified matched resize: {plan['existing_outputs']} original + {plan['synthetic_count']} additional = {len(combined)}")


def status(batch):
    plan = core.read_json(batch / "batch_plan.json")
    refs = core.read_csv(batch / "references/input_manifest.csv", ("reference_id", "repeat_checks"))
    records, done = [], set()
    for shard in (0, 1):
        saved = [core.read_json(p) for p in (batch / f"generated/shard-{shard}/records").glob("ref*.json")]
        records.extend(saved)
        done.update(r["reference_id"] for r in saved)
        print(f"Additional shard {shard}: {sum(len(r['outputs']) for r in saved)}/{plan['synthetic_count']//2}")
    new = sum(len(r["outputs"]) for r in records)
    print(f"MATCHED resize: {plan['existing_outputs']} retained + {new} new = {plan['existing_outputs']+new}/{plan['matched_total']}")
    if records:
        recent = sorted(records, key=lambda r: r["completed_at"])[-12:]
        rate = sum(r["seconds"] for r in recent)/sum(r["passes"] for r in recent)
        work = sum(11 if r["repeat_checks"] == "True" else 8 for r in refs if r["reference_id"] not in done)
        print(f"Estimated remaining for BOTH additional shards: {timedelta(seconds=round(work*rate))}")
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(recent[-1]["completed_at"])).total_seconds()
        print(f"Last source receipt {age:.0f} seconds ago; estimate assumes continued running, excludes QC/transfer")
    else:
        print("Waiting for first completed original source (six new variants) to estimate speed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("parent", "crop", "out"):
        prep.add_argument("--"+name, type=Path, required=True)
    for mode in ("validate", "status", "verify"):
        s = sub.add_parser(mode); s.add_argument("--batch", type=Path, required=True)
    w = sub.add_parser("worker")
    for name in ("batch", "upstream", "weights"):
        w.add_argument("--"+name, type=Path, required=True)
    w.add_argument("--shard", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    if a.command == "prepare": prepare(a)
    elif a.command == "worker": worker(a)
    elif a.command == "verify": verify(a.batch.resolve())
    elif a.command == "status": status(a.batch.resolve())
    else:
        plan, refs = validate(a.batch.resolve())
        print(f"Validated SAME {len(refs)} originals; no new tile IDs; {plan['synthetic_count']} additional / {plan['matched_total']} matched total")


if __name__ == "__main__":
    main()
