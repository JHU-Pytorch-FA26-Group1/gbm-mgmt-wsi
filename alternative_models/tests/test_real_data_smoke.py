"""Real HDF5/PNG reads and metadata gates; tiny CPU encoder, identity stain fixture."""
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch

from alternative_models import real_data_smoke as smoke
from alternative_models.prepare import sha, write_json
from alternative_models.tests.test_engine import tiny


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class RealData(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base/"repo"
        (self.repo/"manifests").mkdir(parents=True)
        self.parts = {}
        self.folds, self.labels = [], []
        self.records = {}
        for half in ("A", "B"):
            root = self.base/f"packages/{half}_qc-v5"
            root.mkdir(parents=True)
            self.parts[half] = root
            rows, people = [], []
            for role in ("train", "validation"):
                for label in (0,1):
                    patient = f"{half}-{role}-{label}"
                    fold = "1" if role == "train" else "0"
                    self.folds.append(dict(source="TCGA", patient_id=patient, fold=fold))
                    self.labels.append(dict(source="TCGA", patient_id=patient, mgmt_status="Methylated" if label else "Unmethylated"))
                    people.append(dict(patient_id=patient))
                    for i in range(3):
                        rgb = np.full((2,2,3), 30+len(rows)*7, dtype=np.uint8)
                        stream = io.BytesIO(); Image.fromarray(rgb).save(stream, format="PNG")
                        rows.append(dict(source="TCGA", patient_id=patient, fold=fold, patch_id=f"{patient}-{i}",
                            stored="True", h5="tiles.h5", h5_index=len(rows), slide_id=patient+"-slide",
                            x_target=i*224, y_target=0, pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                        self.records[rows[-1]["patch_id"]] = stream.getvalue()
            patient = f"{half}-external"
            self.folds.append(dict(source="CPTAC", patient_id=patient, fold="external"))
            self.labels.append(dict(source="CPTAC", patient_id=patient, mgmt_status="Unmethylated"))
            people.append(dict(patient_id=patient))
            write_csv(root/"patient_index.csv", people)
            write_csv(root/"modeling_patches.csv", rows)
            with h5py.File(root/"tiles.h5", "w") as handle:
                data=handle.create_dataset("tiles", (len(rows),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for row in rows:
                    data[row["h5_index"]] = np.frombuffer(self.records[row["patch_id"]], dtype=np.uint8)
        write_csv(self.repo/"manifests/patient_folds.csv", self.folds)
        write_csv(self.repo/"manifests/patient_labels.csv", self.labels)
        for half,root in self.parts.items():
            files = {name:dict(bytes=(root/name).stat().st_size, sha256=sha(root/name))
                     for name in ("modeling_patches.csv", "patient_index.csv", "tiles.h5")}
            write_json(root/"package_manifest.json", dict(subset=half, complete=True, tiling_version="fixture",
                config_sha256="fixture", code_version=7, policy_sha256="fixture",
                folds_sha256=sha(self.repo/"manifests/patient_folds.csv"), files=files))
        self.args = SimpleNamespace(base=self.base, repo=self.repo, out=self.base/"inventory.json")
        smoke.inventory(self.args)
        self.package = self.base/"packages/AB_qc-v5"

    def test_combination_is_metadata_only_complete_and_idempotently_inspectable(self):
        report = json.loads(self.args.out.read_text())
        self.assertEqual(report["tiles"], 24)
        self.assertEqual(report["patients_with_tiles"], {"TCGA":8})
        self.assertEqual(report["patients_without_model_tiles"], ["A-external", "B-external"])
        self.assertFalse((self.package/"tiles.h5").exists())
        self.assertEqual(smoke.inspect(self.package, self.repo)["input_sha256"], report["input_sha256"])

    def test_selection_spans_both_halves_classes_and_roles_without_cptac(self):
        record = smoke.inspect(self.package, self.repo)
        selected = smoke.choose(record, 0, 42)
        self.assertEqual(len(selected), 8)
        self.assertEqual({(r["half"],r["label"],r["role"]) for r in selected.values()},
                         {(h,y,s) for h in ("A","B") for y in (0,1) for s in ("train","validation")})
        self.assertTrue(all(r["source"] == "TCGA" for r in selected.values()))
        rows = smoke.selected_rows(self.package, selected, 2, 42)
        self.assertTrue(all(len(r) == 2 and len({x["tile_id"] for x in r}) == 2 for r in rows.values()))

    def test_real_hdf5_pixels_read_from_both_parts_and_corruption_is_refused(self):
        import loader
        selected = smoke.choose(smoke.inspect(self.package,self.repo), 0, 0)
        rows = smoke.selected_rows(self.package,selected,1,0)
        with loader.TileReader(self.package) as reader:
            for patient, tiles in rows.items():
                image = reader.read(tiles[0]["real_row"])
                self.assertEqual(image.shape, (2,2,3))
                bad = dict(tiles[0]["real_row"], pixel_sha256="bad")
                with self.assertRaisesRegex(ValueError,"pixels differ"):
                    reader.read(bad)

    def test_changed_part_manifest_is_refused(self):
        path = self.parts["B"]/"package_manifest.json"
        path.write_text(path.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"manifest.json changed"):
            smoke.inspect(self.package,self.repo)

    def test_smoke_receipts_remain_invalid_on_failure(self):
        from alternative_models import engine
        args = SimpleNamespace(package=self.package, repo=self.repo, out=self.base/"smoke",
            fold=0, seed=0, backbone="convnext_v2_t", reference_patch_id=None,
            device="cuda:0", tiles_per_patient=2, tile_chunk=2)
        with patch.object(torch.cuda,"is_available",return_value=True), patch.object(engine,"runtime",return_value={}), \
             patch.object(engine,"build",side_effect=RuntimeError("fixture import failure")):
            with self.assertRaisesRegex(RuntimeError,"fixture import failure"):
                smoke.smoke(args)
        receipt=json.loads((args.out/"receipt.json").read_text())
        self.assertFalse(receipt["complete"] or receipt["valid_for_results"])

    def test_both_stages_read_real_fixture_pixels_and_report_patient_predictions(self):
        import loader
        from alternative_models import engine
        class IdentityStain:
            def transform(self, image):
                return image
        args = SimpleNamespace(package=self.package, repo=self.repo, out=self.base/"smoke",
            fold=0, seed=0, backbone="convnext_v2_t", reference_patch_id="A-train-0-0",
            device="cpu", tiles_per_patient=3, tile_chunk=2)
        with patch.object(engine,"build",tiny), patch.object(loader,"fit_fold_normalizer",return_value=(IdentityStain(),{})):
            smoke.smoke(args)
        receipt=json.loads((args.out/"receipt.json").read_text())
        self.assertTrue(receipt["complete"])
        self.assertFalse(receipt["valid_for_results"])
        for stage in ("frozen","finetune"):
            report=json.loads((args.out/f"{stage}_patient_metrics.json").read_text())
            self.assertEqual(report["patient_losses"], 4)
            self.assertEqual(report["optimizer_updates"], 4)
            self.assertEqual(report["patients"], 4)
            self.assertEqual(report["training_tiles"], 12)
            self.assertFalse(report["valid_for_results"])
        with (args.out/"development_predictions.csv").open() as handle:
            predictions=list(csv.DictReader(handle))
        self.assertEqual(len(predictions), 8)
        self.assertTrue(all("validation" in row["patient_id"] for row in predictions))


if __name__ == "__main__":
    unittest.main()
