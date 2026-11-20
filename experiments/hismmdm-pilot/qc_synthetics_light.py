"""Pixel-only synthetic screening using the repository's existing QC rules."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--batch", type=Path, required=True)
p.add_argument("--shard", type=int, choices=(0, 1), required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--blur-reject-below", type=float, choices=(10, 20, 50, 100), default=100)
a = p.parse_args()
protocol = {100: "synthetic-light-v1", 50: "synthetic-light-v2", 20: "synthetic-light-v3", 10: "synthetic-light-v4"}[a.blur_reject_below]
repo, batch = a.repo.resolve(), a.batch.resolve()
sys.path.insert(0, str(repo / "data_prep"))

import common
import qc
import slides
import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

folder = batch / "generated" / f"shard-{a.shard}"
manifest = folder / "generation_manifest.csv"
done = json.loads((folder / "completed.json").read_text())
require(common.sha256_file(manifest) == done["manifest_sha256"],
        "Manifest checksum mismatch")
rows = common.read_rows(manifest)
require(len(rows) == done["outputs"] > 0, "Wrong number of generated images")
require(len({r["variant_id"] for r in rows}) == len(rows),
        "Duplicate variant IDs")

plan = json.loads((batch / "batch_plan.json").read_text())
folds_path = batch / "references" / "patient_folds.csv"
folds = common.read_folds(folds_path)
policy_path = repo / "configs" / "qc_policy.yaml"
tiling_path = repo / "configs" / "tiling.yaml"
policy, policy_sha, policy_text = qc.load_policy(policy_path)
cfg, cfg_sha, cfg_text = common.load_tiling_config(tiling_path)

require(policy["version"] == "qc-v5" and policy["frozen"]
        and policy["sources"] == ["TCGA"],
        "Expected frozen TCGA qc-v5; inspect any newer policy before proceeding")
require(cfg["version"] == "tiling-v7" and cfg["tile_px"] == 224,
        "Expected tiling-v7 / 224 pixels")

# In-memory derivative only. The repository policy file is never modified.
effective = {
    **policy, "grandqc": False, "frozen": False,
    "version": protocol,
    "blur_reject_below": a.blur_reject_below,
    "notes": f"Pixel-only derivative of TCGA qc-v5; synthetic blur reject below {a.blur_reject_below:g}; other pixel cutoffs unchanged; no new GrandQC inference or slide/edge rules."
}

results = []
for i, row in enumerate(rows, 1):
    require(row["source"] == "TCGA", "Non-TCGA source")
    require(str(row["workload_shard"]) == str(a.shard),
            "Shard assignment mismatch")
    assigned = folds.get(("TCGA", row["patient_id"]))
    require(assigned == str(row["assigned_fold"])
            and assigned != str(plan["held_out_fold"])
            and str(row["held_out_fold"]) == str(plan["held_out_fold"]),
            "Fold membership mismatch")

    for field, checksum in [
        ("output_path", "output_sha256"),
        ("classifier_path", "classifier_sha256")
    ]:
        path = (folder / row[field]).resolve()
        require(path.is_relative_to(folder), "Image path escapes shard folder")
        require(common.sha256_file(path) == row[checksum],
                f"Image checksum mismatch: {path}")

    with Image.open(folder / row["classifier_path"]) as im:
        require(im.mode == "RGB" and im.size == (224, 224),
                "Expected existing RGB 224-pixel classifier image")
        scores = slides.tile_scores(np.asarray(im), cfg)

    decision, reasons = qc.decide(scores, effective)
    results.append({
        **row, **scores,
        "pixel_qc_decision": decision,
        "pixel_qc_reasons": reasons,
        "synthetic_grandqc": "not_run"
    })
    if i % 200 == 0:
        print(f"Checked {i}/{len(rows)} images", flush=True)

a.out.mkdir(parents=True, exist_ok=False)
tables = {
    "decisions.csv": results,
    "accepted.csv": [
        r for r in results if r["pixel_qc_decision"] == "accept"
    ],
    "review_queue.csv": [
        r for r in results if r["pixel_qc_decision"] != "accept"
    ]
}
for name, selected in tables.items():
    with (a.out / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(selected)

(a.out / "parent_policy.yaml").write_text(policy_text)
(a.out / "tiling.yaml").write_text(cfg_text)

counts = dict(Counter(r["pixel_qc_decision"] for r in results))
by_class = {
    label: dict(Counter(
        r["pixel_qc_decision"] for r in results
        if r["mgmt_patient_label"] == label
    ))
    for label in sorted({r["mgmt_patient_label"] for r in results})
}
code = [
    repo / "data_prep" / name
    for name in ("qc.py", "slides.py", "common.py", "grandqc.py", "download.py")
]
code.append(Path(__file__).resolve())

receipt = dict(
    protocol=protocol, complete=True, shard=a.shard,
    command=sys.argv,
    created_at=datetime.now(timezone.utc).isoformat(),
    batch=str(batch), images=len(results),
    counts=counts, counts_by_source_mgmt_class=by_class,
    parent_policy_sha256=policy_sha, tiling_sha256=cfg_sha,
    effective_policy=effective,
    blur_override=dict(parent_cutoff=policy["blur_reject_below"], effective_cutoff=a.blur_reject_below),
    manifest_sha256=done["manifest_sha256"],
    completed_sha256=common.sha256_file(folder / "completed.json"),
    batch_run_sha256=common.sha256_file(folder / "batch_run.json"),
    batch_plan_sha256=common.sha256_file(batch / "batch_plan.json"),
    folds_sha256=common.sha256_file(folds_path),
    repo_commit=subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    code_sha256={str(path): common.sha256_file(path) for path in code},
    output_sha256={
        name: common.sha256_file(a.out / name) for name in tables
    },
    runtime=dict(
        python=sys.version, numpy=np.__version__,
        opencv=cv2.__version__, pillow=pillow_version
    ),
    omitted=[
        "new GrandQC inference",
        "slide flags and signoffs",
        "edge-specific decisions"
    ],
    image_policy="Existing 224-pixel classifier images; no resizing, modification or deletion by QC",
    training_eligibility="Unchanged; pixel acceptance does not certify MGMT biology or authorize training"
)
(a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"counts": counts, "by_class": by_class}, indent=2))
print(f"QC results: {a.out.resolve()}")
