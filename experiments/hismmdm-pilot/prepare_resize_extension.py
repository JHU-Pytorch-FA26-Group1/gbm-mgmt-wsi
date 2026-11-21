#!/usr/bin/env python3
"""Prepare additional resize-arm sources without modifying existing runs or samplers."""
import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import re
import tempfile

import his_mmdm_batch as legacy
import his_mmdm_he_augment as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


@contextmanager
def exclude_sources(excluded, audit):
    """Filter only this process's selector input; reuse legacy patient allocation."""
    original = legacy.choose_sources

    def choose(tiles, *args, **kwargs):
        remaining = tiles.loc[~tiles.patch_id.isin(excluded)].copy()
        audit.append(dict(eligible_before=len(tiles), eligible_after=len(remaining),
                          excluded_candidates=len(tiles) - len(remaining)))
        selected, allocation = original(remaining, *args, **kwargs)
        require(not ({r["patch_id"] for r in selected} & excluded), "Selector reused an excluded source")
        return selected, allocation

    legacy.choose_sources = choose
    try:
        yield
    finally:
        legacy.choose_sources = original


def verify_extension(batch):
    plan, rows = legacy.validate_batch(batch)
    receipt = core.read_json(batch / "extension_receipt.json")
    require(receipt["complete"] and receipt["protocol"] == "resize-new-training-tiles-v1",
            "Missing completed source-exclusion receipt")
    require(plan["extension_receipt_sha256"] == core.digest(batch / "extension_receipt.json"),
            "Extension receipt checksum mismatch")
    excluded_path = batch / "references/excluded_parent_sources.csv"
    core.require_hash(excluded_path, receipt["excluded_sources_sha256"])
    excluded_rows = core.read_csv(excluded_path, ("patch_id",))
    core.unique_by(excluded_rows, "patch_id")
    excluded = {r["patch_id"] for r in excluded_rows}
    require(len(excluded) == receipt["excluded_sources"], "Excluded-source count mismatch")
    require(not ({r["patch_id"] for r in rows} & excluded), "New selection overlaps original selection")
    require(len(rows) == receipt["new_sources"] and plan["synthetic_count"] == receipt["planned_variants"],
            "New source/output count mismatch")
    parent = Path(receipt["parent_batch"])
    core.require_hash(parent / "batch_plan.json", receipt["parent_plan_sha256"])
    core.require_hash(parent / "references/input_manifest.csv", receipt["parent_manifest_sha256"])
    core.require_hash(parent / "references/references.json", receipt["parent_references_sha256"])
    parent_plan, parent_rows = legacy.validate_batch(parent)
    require(excluded == {r["patch_id"] for r in parent_rows}, "Exclusion list differs from parent references")
    require(not ({r["reference_id"] for r in rows} & {r["reference_id"] for r in parent_rows}),
            "New reference IDs collide with parent reference IDs")
    require(plan["seed"] != parent_plan["seed"] and plan["held_out_fold"] == parent_plan["held_out_fold"],
            "Expected a new seed and the same held-out fold")
    folds = core.training_ids(batch / "references/patient_folds.csv", plan["held_out_fold"])
    require(all(r["source"] == "TCGA" and r["patient_id"] in folds
                and r["assigned_fold"] != str(plan["held_out_fold"]) for r in rows),
            "Non-training source in extension")
    return plan, rows, receipt


def prepare(args):
    parent = args.parent.resolve()
    require(not args.out.exists(), "Output already exists; use validate/submit to reuse it")
    parent_plan, parents = legacy.validate_batch(parent)
    require(parent_plan["protocol"] == "frozen-hmu1st-he-resize-v1", "Expected the original resize-arm parent")
    require(hasattr(core, "label_status"), "Use the shared core from the working batch bundle")
    parent_prep = core.read_json(parent / "references/references.json")
    fingerprints = dict(parent_plan_sha256=core.digest(parent / "batch_plan.json"),
                        parent_manifest_sha256=core.digest(parent / "references/input_manifest.csv"),
                        parent_references_sha256=core.digest(parent / "references/references.json"))
    seed = parent_plan["seed"] + 1 if args.seed is None else args.seed
    require(seed >= 0 and seed != parent_plan["seed"], "Use a distinct nonnegative sampling seed")
    excluded = {r["patch_id"] for r in parents}
    require(len(excluded) == len(parents), "Parent source IDs are not unique")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        staged = Path(temporary) / "batch"
        # The original preparer performs all package, fold, score and pixel checks.
        # Its temporary latest_batch.json also stays inside this staging directory.
        prepare_args = argparse.Namespace(repo=args.repo, root=args.root, package=args.package,
            labels=args.labels, out=staged, fold=parent_plan["held_out_fold"], subset="A",
            sources_per_class=parent_plan["sources_per_class"],
            verify_per_class=parent_plan["verify_per_class"], seed=seed)
        audit = []
        with exclude_sources(excluded, audit):
            legacy.prepare(prepare_args)
        require(len(audit) == 1, "Unexpected number of legacy selector calls; inspect before proceeding")
        plan, rows = legacy.validate_batch(staged)
        prep = core.read_json(staged / "references/references.json")
        for key in ("folds", "tiling", "settings", "modeling", "labels"):
            require(prep["inputs_sha256"][key] == parent_prep["inputs_sha256"][key],
                    f"Parent and extension use different {key}; inspect before preparing")
        require(len(rows) == len(parents) and not ({r["patch_id"] for r in rows} & excluded),
                "Extension reused old sources or changed source budget")
        require(all(re.fullmatch(r"ref\d{2,6}", r["reference_id"]) for r in parents), "Unexpected parent reference IDs")
        start_id = max(int(r["reference_id"][3:]) for r in parents) + 1
        require(start_id > len(rows) and start_id + len(rows) <= 1000000,
                "Cannot assign disjoint six-digit reference IDs safely")
        for number, row in enumerate(rows, start_id):
            new_ref = f"ref{number:06d}"
            relative = f"originals/{new_ref}.png"
            require(not (staged / "references" / relative).exists(), "Reference destination already exists")
            (staged / "references" / row["image_path"]).rename(staged / "references" / relative)
            row["reference_id"], row["image_path"] = new_ref, relative
        core.save_csv(staged / "references/input_manifest.csv", rows)
        prep["manifest_sha256"] = core.digest(staged / "references/input_manifest.csv")
        legacy.atomic_json(staged / "references/references.json", prep)
        plan["references_sha256"] = core.digest(staged / "references/references.json")
        excluded_rows = [{k: r[k] for k in ("reference_id", "source", "patient_id", "slide_id", "patch_id")}
                         for r in parents]
        excluded_path = staged / "references/excluded_parent_sources.csv"
        core.save_csv(excluded_path, excluded_rows)
        receipt = dict(protocol="resize-new-training-tiles-v1", complete=True,
            parent_batch=str(parent), **fingerprints, excluded_sources=len(excluded),
            excluded_sources_sha256=core.digest(excluded_path), new_sources=len(rows),
            planned_variants=plan["synthetic_count"], source_overlap=0, sampling_seed=seed,
            reference_id_start=start_id,
            held_out_fold=plan["held_out_fold"], selector_audit=audit[0],
            source_counts=dict(Counter(r["mgmt_patient_label"] for r in rows)),
            patients_per_class={label: len({r["patient_id"] for r in rows if r["mgmt_patient_label"] == label})
                               for label in legacy.LABELS},
            preparer_sha256=core.digest(Path(__file__)), legacy_batch_sha256=core.digest(Path(legacy.__file__)),
            shared_core_sha256=core.digest(Path(core.__file__)), created_at=core.stamp(),
            review_status="pending", training_eligible=False)
        legacy.atomic_json(staged / "extension_receipt.json", receipt)
        plan["extension_receipt_sha256"] = core.digest(staged / "extension_receipt.json")
        legacy.atomic_json(staged / "batch_plan.json", plan)
        verify_extension(staged)
        staged.rename(args.out)
    print(f"Prepared {len(rows)} NEW training sources for {plan['synthetic_count']} resize variants.")
    print(f"Original source overlap: 0; held-out fold: {plan['held_out_fold']}; seed: {seed}")
    print(f"Output: {args.out.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare")
    for name in ("repo", "root", "package", "labels", "parent", "out"):
        make.add_argument("--" + name, type=Path, required=True)
    make.add_argument("--seed", type=int)
    check = sub.add_parser("validate")
    check.add_argument("--batch", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a)
    else:
        plan, rows, receipt = verify_extension(a.batch.resolve())
        print(f"Validated {len(rows)} new sources; {plan['synthetic_count']} planned outputs; overlap 0")


if __name__ == "__main__":
    main()
