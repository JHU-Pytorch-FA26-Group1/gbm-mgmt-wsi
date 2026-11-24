"""Half B ZIP integrity, ownership and no-overwrite fixtures; no GPU/model imports."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import stat
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from alternative_models import half_b_package as pre
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("half_b_source_split", REPO / "data_prep/split.py")
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)

def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class HalfB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original = self.root / "A"; self.original.mkdir()
        frozen = json.loads((REPO / "manifests/collection_manifest.json").read_text())
        owners = {r["patient_id"]: r["half"] for r in split.build_patient_split(frozen["files"])}
        labels = {r["patient_id"]: r for r in pre.read_rows(REPO / "manifests/patient_labels.csv")}
        folds = pre.read_rows(REPO / "manifests/patient_folds.csv")
        self.people = [next(r for r in folds if r["source"] == "TCGA" and owners[r["patient_id"]] == "B"
            and r["fold"] != "0" and labels[r["patient_id"]]["mgmt_status"] == label)
            for label in ("Methylated", "Unmethylated")]
        self.model = [dict(r, patch_id=f"patch{i}", stored="True", h5="tiles/test.h5", role="tissue")
                      for i, r in enumerate(self.people)]
        self.payloads = {"modeling_patches.csv": csv_bytes(self.model),
            "patient_index.csv": b"patient_id\n", "slide_status.csv": b"slide_id\n",
            "export_settings.json": b"{}", "tiles/test.h5": b"fixture bytes; no HDF5 reads"}
        self.manifest = dict(kind="package", subset="B", complete=True, tiling_version="tiling-v7",
            config_sha256=pre.sha256(REPO / "configs/tiling.yaml"), code_version=7,
            policy_version="qc-v5+qc-v5-cptac", policy_sha256="fixture policy hash",
            folds_sha256=pre.sha256(REPO / "manifests/patient_folds.csv"), counts=dict(tiles=2, slides=1))
        (self.original / "package_manifest.json").write_text(json.dumps(dict(self.manifest, subset="A")))
        self.path = self.root / "B.zip"
        self.write()

    def write(self, extra=None, corrupt=None):
        self.manifest["files"] = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                                  for name, data in self.payloads.items()}
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("B_qc-v5/package_manifest.json", json.dumps(self.manifest))
            for name, data in self.payloads.items(): z.writestr("B_qc-v5/"+name, corrupt if name == "tiles/test.h5" and corrupt is not None else data)
            if extra: z.writestr(extra, b"unsafe")

    def test_metadata_population_and_single_pass_extraction(self):
        with zipfile.ZipFile(self.path) as z:
            m, prefix, planned = pre.catalog(z, self.original, REPO)
            result = pre.population(z, prefix, m, REPO)
            self.assertEqual(result["eligible_fold0_training_patients"], dict(Methylated=1, Unmethylated=1))
            self.assertFalse(result["enough_for_1000_distinct_sources_per_class"])
            out = self.root / "extracted"
            pre.extract_verified(z, prefix, m, planned, out)
            for name, data in self.payloads.items(): self.assertEqual((out / name).read_bytes(), data)
            with self.assertRaisesRegex(ValueError, "already exists"):
                pre.extract_verified(z, prefix, m, planned, out)

    def test_zip_paths_and_incompatible_policies_are_refused(self):
        self.write(extra="../outside")
        with zipfile.ZipFile(self.path) as z:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                pre.catalog(z, self.original, REPO)
        self.manifest["policy_sha256"] = "different"; self.write()
        with zipfile.ZipFile(self.path) as z:
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                pre.catalog(z, self.original, REPO)

    def test_same_size_corrupted_payload_never_publishes_output(self):
        self.write(corrupt=b"x"*len(self.payloads["tiles/test.h5"]))
        out = self.root / "bad"
        with zipfile.ZipFile(self.path) as z:
            m, prefix, planned = pre.catalog(z, self.original, REPO)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                pre.extract_verified(z, prefix, m, planned, out)
        self.assertFalse(out.exists())

    def test_half_a_patient_in_b_is_refused(self):
        frozen = json.loads((REPO / "manifests/collection_manifest.json").read_text())
        owners = split.build_patient_split(frozen["files"])
        patient = next(r["patient_id"] for r in owners if r["half"] == "A" and r["source"] == "TCGA")
        self.model[0]["patient_id"] = patient
        self.payloads["modeling_patches.csv"] = csv_bytes(self.model); self.write()
        with zipfile.ZipFile(self.path) as z:
            m, prefix, _ = pre.catalog(z, self.original, REPO)
            with self.assertRaisesRegex(ValueError, "Non-Half-B"):
                pre.population(z, prefix, m, REPO)

    def test_symlink_member_is_refused_before_extraction(self):
        with zipfile.ZipFile(self.path, "a") as archive:
            link = zipfile.ZipInfo("B_qc-v5/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "../../outside")
        with zipfile.ZipFile(self.path) as archive:
            with self.assertRaisesRegex(ValueError, "symlinks"):
                pre.catalog(archive, self.original, REPO)

    def test_insufficient_space_does_not_publish_output(self):
        out = self.root / "no-space"
        with zipfile.ZipFile(self.path) as archive:
            m, prefix, planned = pre.catalog(archive, self.original, REPO)
            with patch.object(pre.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, "Insufficient free space"):
                    pre.extract_verified(archive, prefix, m, planned, out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
