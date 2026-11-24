#!/usr/bin/env python3
"""Inspect a portable Half B ZIP; optionally extract/hash-verify it. Never submit jobs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

SHARED = ("tiling_version", "config_sha256", "code_version", "policy_sha256", "folds_sha256")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(name and path.parts and path.as_posix() == name.rstrip("/")
            and not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and ":" not in path.parts[0], f"Unsafe archive/manifest path: {name}")
    return path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def catalog(archive, a_package, repo):
    infos = archive.infolist()
    files = {}
    for info in infos:
        safe_name(info.filename)
        require(info.filename not in files, "Duplicate ZIP member")
        require(not stat.S_ISLNK(info.external_attr >> 16), "ZIP symlinks are not supported")
        files[info.filename] = info
    manifests = [name for name, info in files.items() if not info.is_dir()
                 and PurePosixPath(name).name == "package_manifest.json"
                 and not name.startswith("__MACOSX/")]
    require(len(manifests) == 1, "Expected exactly one portable package_manifest.json")
    member = manifests[0]
    require(files[member].file_size <= 2*1024*1024, "Unexpectedly large package manifest")
    manifest = json.loads(archive.read(member))
    require(manifest["kind"] == "package" and manifest["subset"] == "B" and manifest["complete"] is True,
            "Expected a complete portable Half B package")
    require(manifest["tiling_version"] == "tiling-v7", "Expected tiling-v7")
    require(set(manifest["policy_version"].split("+")) == {"qc-v5", "qc-v5-cptac"}, "Expected the two qc-v5 policies")
    original = json.loads((a_package / "package_manifest.json").read_text())
    require(original["subset"] == "A" and original["complete"] is True, "Half A comparison package is incomplete/wrong")
    require(all(manifest[k] == original[k] for k in SHARED), "A/B tiling, code, policy or fold hashes differ")
    require(sha256(repo / "manifests/patient_folds.csv") == manifest["folds_sha256"], "Repository fold map differs")
    require(sha256(repo / "configs/tiling.yaml") == manifest["config_sha256"], "Repository tiling config differs")
    prefix = member[:-len("package_manifest.json")]
    planned = {}
    for name, record in manifest["files"].items():
        safe_name(name)
        require(name != "package_manifest.json" and name == PurePosixPath(name).as_posix(), "Noncanonical manifest filename")
        require(type(record["bytes"]) is int and record["bytes"] >= 0, "Invalid manifest file size")
        require(len(record["sha256"]) == 64 and all(c in "0123456789abcdef" for c in record["sha256"]), "Invalid SHA-256")
        info = files.get(prefix + name)
        require(info is not None and not info.is_dir() and info.file_size == record["bytes"], f"Missing/wrong-size member: {name}")
        planned[name] = record
    require({"modeling_patches.csv", "patient_index.csv", "slide_status.csv", "export_settings.json"} <= planned.keys(),
            "Package metadata incomplete")
    return manifest, prefix, planned


class HashReader(io.RawIOBase):
    """Hash the exact CSV bytes while csv.DictReader consumes a text wrapper."""
    def __init__(self, source):
        self.source, self.hash, self.bytes = source, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.source.read(len(buffer))
        buffer[:len(data)] = data
        self.hash.update(data)
        self.bytes += len(data)
        return len(data)


def population(archive, prefix, manifest, repo):
    labels = {r["patient_id"]: r for r in read_rows(repo / "manifests/patient_labels.csv")}
    folds = {r["patient_id"]: r for r in read_rows(repo / "manifests/patient_folds.csv")}
    spec = importlib.util.spec_from_file_location("half_b_split", repo / "data_prep/split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = json.loads((repo / "manifests/collection_manifest.json").read_text())
    owners = {r["patient_id"]: r["half"] for r in module.build_patient_split(frozen["files"])}
    candidates, patients, totals, identities = Counter(), defaultdict(set), Counter(), {}
    seen = set()
    with archive.open(prefix + "modeling_patches.csv") as handle:
        raw = HashReader(handle)
        reader = csv.DictReader(io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8-sig", newline=""))
        for row in reader:
            p, patch = row["patient_id"], row["patch_id"]
            require(p in labels and p in folds and owners.get(p) == "B", f"Non-Half-B/unknown patient: {p}")
            require(row["source"] == labels[p]["source"] == folds[p]["source"] and row["fold"] == folds[p]["fold"],
                    f"Source/fold mismatch: {p}")
            require(patch not in seen, f"Duplicate patch: {patch}")
            seen.add(patch)
            require(row["stored"] == "True" and row["h5"] in manifest["files"], "Expected packaged, stored tile")
            require(row["role"] in {"tissue", "edge"}, "Unexpected packaged tissue/edge role")
            totals[row["source"]] += 1
            identities[p] = row["source"]
            if row["source"] == "TCGA" and row["fold"] != "0" and row["role"] == "tissue":
                label = labels[p]["mgmt_status"]
                require(label in {"Methylated", "Unmethylated"}, "Unknown MGMT status")
                candidates[label] += 1
                patients[label].add(p)
        expected = manifest["files"]["modeling_patches.csv"]
        require(raw.hash.hexdigest() == expected["sha256"] and raw.bytes == expected["bytes"],
                "Modeling table content differs from package hash")
    require(len(seen) == manifest["counts"]["tiles"], "Modeling row count differs from package tile count")
    return dict(packaged_tiles=dict(totals), patients_with_tiles=dict(Counter(identities.values())),
                eligible_fold0_training_tissue_tiles=dict(candidates),
                eligible_fold0_training_patients={label: len(values) for label, values in patients.items()},
                enough_for_1000_distinct_sources_per_class=all(candidates[k] >= 1000 for k in ("Methylated", "Unmethylated")))


def completion(folder, expected):
    path = folder / "completed.json"
    if not path.exists():
        return dict(complete=False, reason="No completion receipt")
    try:
        done = json.loads(path.read_text())
        require(done["outputs"] == expected, "Wrong completed output count")
        require(sha256(folder / "generation_manifest.csv") == done["manifest_sha256"], "Generation manifest changed")
        return dict(complete=True, outputs=expected, scope="Completion count and CSV hash only; image files not rehashed")
    except (OSError, ValueError, KeyError) as error:
        return dict(complete=False, reason=str(error))


def half_a_status(base):
    pilot = base / "GBM_MGMT/hismmdm-pilot"
    crop = [completion(pilot / f"fold-0-corners-16000/generated/shard-{i}", 8000) for i in (0, 1)]
    resize = [completion(pilot / f"fold-0-balanced-4000/generated/shard-{i}", 2000) for i in (0, 1)]
    resize += [completion(pilot / f"fold-0-resize-matched-additional12000/generated/shard-{i}", 6000) for i in (0, 1)]
    combined = pilot / "fold-0-resize-matched-additional12000/completed.json"
    combined_ok = False
    if combined.exists():
        try:
            done = json.loads(combined.read_text())
            combined_ok = done.get("matched_total") == 16000 and sha256(combined.parent / "matched_resize_16000.csv") == done.get("manifest_sha256")
        except (OSError, ValueError, KeyError):
            combined_ok = False
    return dict(corner_shards=crop, resize_original_and_additional_shards=resize,
                resize_combined_receipt_valid=combined_ok,
                generation_completion_records_ready=all(r["complete"] for r in crop+resize) and combined_ok,
                gpu_release="Not verified: inspect Run:ai workloads; absence of a job alone does not prove successful completion")


def extract_verified(archive, prefix, manifest, planned, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), "Output already exists; do not overwrite it")
    needed = sum(r["bytes"] for r in planned.values())
    require(shutil.disk_usage(out.parent).free >= needed*1.02 + 512*1024**2, "Insufficient free space for extraction")
    with tempfile.TemporaryDirectory(prefix="B_qc-v5.extract-", dir=out.parent) as temp:
        stage = Path(temp) / "package"
        stage.mkdir()
        for i, (name, record) in enumerate(planned.items(), 1):
            path = stage.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            value, total = hashlib.sha256(), 0
            with archive.open(prefix + name) as source, path.open("xb") as target:
                for chunk in iter(lambda: source.read(1024*1024), b""):
                    target.write(chunk)
                    value.update(chunk)
                    total += len(chunk)
            require(total == record["bytes"] and value.hexdigest() == record["sha256"], f"Content/hash mismatch: {name}")
            if i % 20 == 0 or i == len(planned):
                print(f"Extracted and hash-verified {i}/{len(planned)} package files", file=sys.stderr, flush=True)
        (stage / "package_manifest.json").write_bytes(archive.read(prefix + "package_manifest.json"))
        require(not out.exists(), "Output appeared during extraction; stopped without overwriting")
        stage.rename(out)
    return dict(path=str(out.resolve()), files_verified=len(planned), bytes_verified=needed,
                scope="Verified against the ZIP's package manifest; no original WSI or clinical-label re-audit")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/workspace/persist/crivas4-microscopy"))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--zip", type=Path)
    p.add_argument("--a-package", type=Path)
    p.add_argument("--extract-to", type=Path, help="Optional NEW folder; verifies hashes while extracting")
    a = p.parse_args()
    path = a.zip or a.base / "B_qc-v5.zip"
    try:
        with zipfile.ZipFile(path) as archive:
            manifest, prefix, planned = catalog(archive, a.a_package or a.base / "packages/A_qc-v5", a.repo)
            pop = population(archive, prefix, manifest, a.repo)
            report = dict(zip=str(path.resolve()), archive_prefix=prefix,
                          expanded_GiB=round(sum(r["bytes"] for r in planned.values()) / 2**30, 2),
                          compatible_with_half_a=True, package_counts=manifest["counts"], population=pop,
                          source_score_sidecars=[i.filename for i in archive.infolist() if i.filename.endswith("_patches.csv")][:10],
                          source_preparation_requirement="Existing preparer needs original mask_fraction/source score CSVs; a normal portable package does not contain them. No values will be invented.",
                          half_a=half_a_status(a.base), jobs_submitted=0)
            if a.extract_to:
                report["extraction"] = extract_verified(archive, prefix, manifest, planned, a.extract_to)
            print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Half B preflight stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
