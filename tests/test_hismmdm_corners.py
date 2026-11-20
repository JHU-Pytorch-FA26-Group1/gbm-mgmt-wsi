"""CPU checks for geometry, patient isolation and resume integrity; no WSI/GPU needed."""
from collections import Counter
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/hismmdm-pilot"))
import his_mmdm_corners as c


class Geometry(unittest.TestCase):
    def test_exact_pixels_and_overlap(self):
        from PIL import Image
        original = Image.new("RGB", (224, 224))
        original.putdata([(x, y, (x + y) % 256) for y in range(224) for x in range(224)])
        crops = {name: c.crop_rgb(original, name) for name in c.CORNERS}
        for name, (x, y) in c.CORNERS.items():
            cropped = crops[name]
            self.assertEqual(cropped.size, (128, 128))
            for px, py in ((0, 0), (127, 127), (53, 81)):
                self.assertEqual(cropped.getpixel((px, py)), original.getpixel((x + px, y + py)))
        self.assertEqual(crops["tl"].crop((96, 0, 128, 128)).tobytes(),
                         crops["tr"].crop((0, 0, 32, 128)).tobytes())
        self.assertEqual(crops["tl"].crop((0, 96, 128, 128)).tobytes(),
                         crops["bl"].crop((0, 0, 128, 32)).tobytes())

    def test_wrong_geometry_is_refused(self):
        from PIL import Image
        for image in (Image.new("RGB", (128, 128)), Image.new("L", (224, 224))):
            with self.assertRaises(ValueError):
                c.crop_rgb(image, "tl")


class ParentLineage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name)
        self.refs = self.parent / "references"
        self.refs.mkdir()
        self.rows = []
        self.folds = []
        self.labels = []
        for p in range(63):
            patient = f"TCGA-01-{p:04d}"
            self.folds.append(dict(patient_id=patient, source="TCGA", fold="1"))
            self.labels.append(dict(patient_id=patient, mgmt="1" if p < 29 else "0"))
        for i in range(2000):
            meth = i < 1000
            patient = f'TCGA-01-{(i % 29) if meth else (29 + i % 34):04d}'
            ref = f"ref{i + 1:06d}"
            self.rows.append(dict(reference_id=ref, patch_id=f"patch{i}", patient_id=patient,
                source="TCGA", assigned_fold="1", slide_id=f"slide{i}", source_mpp="0.5",
                image_path=f"originals/{ref}.png", workload_shard=str(i % 2), repeat_checks="False",
                mgmt_patient_label="methylated" if meth else "unmethylated", qc_defect_flags="",
                source_sha256="1" * 64, pixel_sha256="2" * 64))

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        c.core.save_csv(self.refs / "patient_folds.csv", self.folds)
        c.core.save_csv(self.refs / "patient_labels.csv", self.labels)
        c.core.save_csv(self.refs / "input_manifest.csv", self.rows)
        c.core.save_csv(self.parent / "patient_allocation.csv", [dict(patient_id="recorded", selected_sources=2000)])
        c.atom(self.refs / "references.json", dict(held_out_fold=0,
            manifest_sha256=c.core.digest(self.refs / "input_manifest.csv"),
            inputs_sha256=dict(folds=c.core.digest(self.refs / "patient_folds.csv"),
                              labels=c.core.digest(self.refs / "patient_labels.csv"))))
        c.atom(self.parent / "batch_plan.json", dict(protocol="frozen-hmu1st-he-resize-v1", source_count=2000, synthetic_count=4000,
            sources_per_class=1000, shard_count=2, variants_per_source=2, eta=0.3, held_out_fold=0, seed=0,
            references_sha256=c.core.digest(self.refs / "references.json"),
            allocation_sha256=c.core.digest(self.parent / "patient_allocation.csv")))

    def test_valid_balanced_training_parent(self):
        self.write()
        _, _, rows = c.parent_inputs(self.parent)
        self.assertEqual(len(rows), 2000)

    def test_held_out_patient_is_refused_even_with_consistent_hashes(self):
        self.folds[0]["fold"] = "0"
        for r in self.rows:
            if r["patient_id"] == self.folds[0]["patient_id"]:
                r["assigned_fold"] = "0"
        self.write()
        with self.assertRaisesRegex(ValueError, "Non-training"):
            c.parent_inputs(self.parent)

    def test_duplicate_patch_is_refused(self):
        self.rows[1]["patch_id"] = self.rows[0]["patch_id"]
        self.write()
        with self.assertRaises(ValueError):
            c.parent_inputs(self.parent)

    def test_unbalanced_shard_is_refused(self):
        self.rows[0]["workload_shard"] = "1"
        self.write()
        with self.assertRaisesRegex(ValueError, "balance"):
            c.parent_inputs(self.parent)

    def test_label_conflict_is_refused(self):
        self.labels[0]["mgmt_status"] = "unmethylated"
        with self.assertRaises(ValueError):
            c.label(self.labels[0])

    def test_metadata_tamper_is_refused(self):
        self.write()
        with (self.refs / "input_manifest.csv").open("a") as f:
            f.write("changed\n")
        with self.assertRaises(ValueError):
            c.parent_inputs(self.parent)

    def test_complete_preparation_and_geometry_tamper(self):
        from PIL import Image
        image = Image.new("RGB", (224, 224))
        image.putdata([(x, y, (x + y) % 256) for y in range(224) for x in range(224)])
        (self.refs / "originals").mkdir()
        first = self.refs / "originals/ref000001.png"
        image.save(first)
        original_hash = c.core.digest(first)
        pixel_hash = c.hashlib.sha256(image.tobytes()).hexdigest()
        for r in self.rows:
            if r["reference_id"] != "ref000001":
                c.shutil.copyfile(first, self.refs / r["image_path"])
            r["source_sha256"], r["pixel_sha256"] = original_hash, pixel_hash
        self.write()
        out = self.parent / "new-crop-arm"
        c.prepare(SimpleNamespace(parent=self.parent, out=out))
        plan, rows = c.validated(out)
        self.assertEqual(len(rows), 8000)
        self.assertEqual(plan["synthetic_count"], 16000)
        self.assertEqual(len(list((out / "references/crops").glob("*.png"))), 8000)
        with self.assertRaises(ValueError):
            c.prepare(SimpleNamespace(parent=self.parent, out=out))
        rows[0]["crop_x"] = "1"
        c.core.save_csv(out / "references/input_manifest.csv", rows)
        plan["manifest_sha256"] = c.core.digest(out / "references/input_manifest.csv")
        c.atom(out / "batch_plan.json", plan)
        with self.assertRaisesRegex(ValueError, "geometry"):
            c.validated(out)


class Replay(unittest.TestCase):
    def test_fixed_groups_and_rng_membership(self):
        rows = []
        for shard in (0, 1):
            for klass in c.LABELS:
                for source in range(4):
                    for corner in c.CORNERS:
                        rows.append(dict(reference_id=f"{shard}-{klass}-{source}-{corner}",
                                         workload_shard=str(shard), mgmt_patient_label=klass,
                                         repeat_checks=str(source == 0)))
        expected = c.grouped(rows, 0, 8)
        random.Random(23).shuffle(rows)
        self.assertEqual(expected, c.grouped(rows, 0, 8))
        self.assertTrue(all(len(group) == 8 for group in expected))
        self.assertTrue(all(Counter(r["mgmt_patient_label"] for r in group)
                            == Counter(dict.fromkeys(c.LABELS, 4)) for group in expected))
        ids = [r["reference_id"] for r in expected[0]]
        self.assertEqual(c.seed_for(0, ids, 1), c.seed_for(0, ids, 1))
        self.assertNotEqual(c.seed_for(0, ids, 1), c.seed_for(0, ids, 2))
        self.assertNotEqual(c.seed_for(0, ids, 1), c.seed_for(0, list(reversed(ids)), 1))

    def test_receipt_requires_matching_membership_and_intact_files(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.assertIsNone(c.checked_record(folder, "group00000", ["ref1"], "signature"))
            (folder / "records").mkdir()
            files, outputs = {}, []
            for v in (1, 2):
                native, classifier, control = f"native{v}.png", f"classifier{v}.png", "control.png"
                for name in (native, classifier, control):
                    (folder / name).write_bytes(name.encode())
                    files[name] = c.core.digest(folder / name)
                outputs.append(dict(variant_id=f"ref1-{v:02d}", output_path=native, output_sha256=files[native],
                    classifier_path=classifier, classifier_sha256=files[classifier],
                    reconstruction_path=control, reconstruction_sha256=files[control]))
            c.atom(folder / "records/group00000.json", dict(input_ids=["ref1"],
                   signature_sha256="signature", file_sha256=files, outputs=outputs))
            self.assertIsNotNone(c.checked_record(folder, "group00000", ["ref1"], "signature"))
            with self.assertRaises(ValueError):
                c.checked_record(folder, "group00000", ["another"], "signature")
            with self.assertRaises(ValueError):
                c.checked_record(folder, "group00000", ["ref1"], "changed-code")
            (folder / "native1.png").write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                c.checked_record(folder, "group00000", ["ref1"], "signature")

    def test_path_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                c.inside(Path(d), "../outside.png")


if __name__ == "__main__":
    unittest.main()
