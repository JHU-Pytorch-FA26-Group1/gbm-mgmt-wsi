"""CPU source-isolation and real package-reader tests; no generator/GPU required."""
from collections import Counter
import hashlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/hismmdm-pilot"))
import prepare_resize_extension as e
import his_mmdm_batch as b
import his_mmdm_he_augment as c


class Exclusion(unittest.TestCase):
    def test_exclusion_before_patient_balancing(self):
        rows = []
        labels = {}
        for klass in b.LABELS:
            for p in range(3):
                patient = f"{klass}-{p}"
                labels[patient] = klass
                for k in range(8):
                    rows.append(dict(patient_id=patient, patch_id=f"{patient}-{k}",
                                     slide_id=patient, x_target=k*224, y_target=0))
        tiles = pd.DataFrame(rows)
        excluded = {r["patch_id"] for r in rows if r["patch_id"].endswith("-0")}
        sys.path.insert(0, str(ROOT / "data_prep"))
        import loader
        original = b.choose_sources
        audit = []
        with e.exclude_sources(excluded, audit):
            selected, _ = b.choose_sources(tiles, labels, 8, 43, 0, loader)
        self.assertIs(b.choose_sources, original)
        self.assertEqual(len(selected), 16)
        self.assertFalse({r["patch_id"] for r in selected} & excluded)
        self.assertEqual(len({r["patient_id"] for r in selected}), 6)
        self.assertEqual(Counter(r["mgmt_patient_label"] for r in selected), Counter(dict.fromkeys(b.LABELS, 8)))
        self.assertEqual(audit[0]["excluded_candidates"], 6)

    def test_exhaustion_refuses_and_restores(self):
        original = b.choose_sources
        tiles = pd.DataFrame([dict(patient_id="p", patch_id="old", slide_id="s", x_target=0, y_target=0)])
        with self.assertRaises(ValueError):
            with e.exclude_sources({"old"}, []):
                b.choose_sources(tiles, {"p": "methylated"}, 8, 43, 0, None)
        self.assertIs(b.choose_sources, original)

    def test_restores_after_an_unrelated_error(self):
        original = b.choose_sources
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with e.exclude_sources(set(), []):
                raise RuntimeError("failure")
        self.assertIs(b.choose_sources, original)


class PackagePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.repo, cls.data, cls.package = cls.root / "repo", cls.root / "data", cls.root / "package"
        (cls.repo / "configs").mkdir(parents=True)
        (cls.repo / "manifests").mkdir()
        (cls.repo / "data_prep").symlink_to(ROOT / "data_prep", target_is_directory=True)
        cls.package.mkdir()
        shutil.copyfile(ROOT / "configs/tiling.yaml", cls.repo / "configs/tiling.yaml")
        modeling, folds, labels, index, files = [], [], [], [], {}
        for p in range(8):
            patient = f"TCGA-AA-{p:04d}" if p < 7 else "C3L-TEST"
            source = "TCGA" if p < 7 else "CPTAC"
            fold = "1" if p < 6 else ("0" if p == 6 else "external")
            folds.append(dict(source=source, patient_id=patient, fold=fold))
            labels.append(dict(patient_id=patient, mgmt=int(p < 3)))
            slide, stored, scores = patient + "-slide", [], []
            for k in range(13):
                pixels = np.random.default_rng(p*100+k).integers(40, 220, (224, 224, 3), dtype=np.uint8)
                stream = io.BytesIO()
                Image.fromarray(pixels).save(stream, format="PNG")
                stored.append(np.frombuffer(stream.getvalue(), dtype=np.uint8))
                patch_id = f"{patient}-p{k}"
                sha = hashlib.sha256(pixels.tobytes()).hexdigest()
                modeling.append(dict(source=source, patient_id=patient, fold=fold, role="tissue" if k < 12 else "edge",
                    slide_id=slide, patch_id=patch_id, pixel_sha256=sha, stored=True,
                    h5=slide+".h5", h5_index=k, x_target=k*224, y_target=p*224))
                scores.append(dict(patch_id=patch_id, pixel_sha256=sha, mask_fraction=.9,
                                   laplacian_variance=150, near_white_fraction=.01, near_black_fraction=0))
            with h5py.File(cls.package / (slide+".h5"), "w") as handle:
                dataset = handle.create_dataset("tiles", (len(stored),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for i, value in enumerate(stored):
                    dataset[i] = value
            score_path = cls.data / "scores" / (slide+".csv")
            score_path.parent.mkdir(exist_ok=True, parents=True)
            c.save_csv(score_path, scores)
            index.append(dict(source=source, slide_id=slide, csv=str(score_path.relative_to(cls.data))))
            path = cls.package / (slide+".h5")
            files[path.name] = dict(bytes=path.stat().st_size, sha256=c.digest(path))
        c.save_csv(cls.repo / "manifests/patient_folds.csv", folds)
        c.save_csv(cls.root / "labels.csv", labels)
        c.save_csv(cls.package / "modeling_patches.csv", modeling)
        path = cls.package / "modeling_patches.csv"
        files[path.name] = dict(bytes=path.stat().st_size, sha256=c.digest(path))
        sys.path.insert(0, str(ROOT / "data_prep"))
        import common
        cfg_hash = common.load_tiling_config(cls.repo / "configs/tiling.yaml")[1]
        c.save_json(cls.package / "package_manifest.json", dict(complete=True, files=files,
            folds_sha256=c.digest(cls.repo / "manifests/patient_folds.csv"), config_sha256=cfg_hash))
        idx = cls.data / "patches/tiling-v7/index_A.csv"
        idx.parent.mkdir(parents=True)
        c.save_csv(idx, index)
        cls.parent = cls.root / "pilot/parent"
        cls.parent.parent.mkdir()
        b.prepare(SimpleNamespace(repo=cls.repo, root=cls.data, package=cls.package, labels=cls.root / "labels.csv",
            out=cls.parent, fold=0, subset="A", sources_per_class=8, verify_per_class=1, seed=42))
        cls.pointer = cls.parent.parent / "latest_batch.json"
        cls.pointer_bytes = cls.pointer.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def new_extension(self, name):
        out = self.root / "pilot" / name
        e.prepare(SimpleNamespace(repo=self.repo, root=self.data, package=self.package,
            labels=self.root / "labels.csv", parent=self.parent, out=out, seed=None))
        return out

    def test_preparation_disjoint_ids_pixels_and_training_folds(self):
        out = self.new_extension("new-tiles")
        plan, rows, receipt = e.verify_extension(out)
        _, parents = b.validate_batch(self.parent)
        self.assertEqual(len(rows), 16)
        self.assertEqual(plan["synthetic_count"], 32)
        self.assertEqual(plan["seed"], 43)
        self.assertEqual(receipt["source_overlap"], 0)
        self.assertEqual(self.pointer.read_bytes(), self.pointer_bytes)
        self.assertFalse({r["patch_id"] for r in rows} & {r["patch_id"] for r in parents})
        self.assertFalse({r["reference_id"] for r in rows} & {r["reference_id"] for r in parents})
        self.assertTrue(all(r["source"] == "TCGA" and r["assigned_fold"] == "1" for r in rows))
        self.assertTrue(all(not r["patch_id"].endswith("p12") for r in rows))
        self.assertEqual(Counter(r["mgmt_patient_label"] for r in rows), Counter(dict.fromkeys(b.LABELS, 8)))
        for row in rows:
            path = out / "references" / row["image_path"]
            self.assertEqual(c.digest(path), row["source_sha256"])
            with Image.open(path) as im:
                self.assertEqual(hashlib.sha256(im.tobytes()).hexdigest(), row["pixel_sha256"])

    def test_exclusion_tampering_is_detected(self):
        out = self.new_extension("tamper")
        with (out / "references/excluded_parent_sources.csv").open("a") as handle:
            handle.write("corrupt\n")
        with self.assertRaises(ValueError):
            e.verify_extension(out)

    def test_existing_output_is_not_overwritten(self):
        out = self.new_extension("existing")
        before = c.digest(out / "batch_plan.json")
        with self.assertRaises(ValueError):
            e.prepare(SimpleNamespace(parent=self.parent, out=out))
        self.assertEqual(c.digest(out / "batch_plan.json"), before)


if __name__ == "__main__":
    unittest.main()
