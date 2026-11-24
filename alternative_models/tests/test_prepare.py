import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from alternative_models.prepare import prepare, sha, write_json


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.package = self.root/"repo", self.root/"package"
        (self.repo/"manifests").mkdir(parents=True)
        self.package.mkdir()
        self.canonical = [("T1", "1", "Methylated"), ("T2", "2", "Unmethylated"),
                          ("V1", "0", "Methylated"), ("V2", "0", "Unmethylated")]
        table(self.repo/"manifests/patient_folds.csv", [dict(source="TCGA", patient_id=p, fold=f) for p,f,_ in self.canonical])
        table(self.repo/"manifests/patient_labels.csv", [dict(source="TCGA", patient_id=p, mgmt_status=s) for p,_,s in self.canonical])
        self.rows = [dict(source="TCGA", patient_id=p, fold=f, patch_id=f"{p}-{i}",
                          slide_id=f"{p}-slide", x_target=i*224, y_target=0, pixel_sha256="a"*64)
                     for p,f,_ in self.canonical for i in range(8)]
        table(self.package/"modeling_patches.csv", self.rows)
        self.fake_loader = SimpleNamespace(load_modeling=lambda *a, **kw: pd.DataFrame(self.rows))
        self.args = SimpleNamespace(repo=self.repo, package=self.package, out=self.root/"plan",
                                    synthetic_inputs=None, fold=0, cap=6, seed=0)

    def synthetic(self, **changes):
        record = dict(patch_id="T1-0", patient_id="T1", source="TCGA", pixel_qc_decision="accept",
                      held_out_fold="0", assigned_fold="1", slide_id="T1-slide", variant_id="ref1-01",
                      mgmt_patient_label="methylated", pixel_sha256="a"*64,
                      classifier_path="variant.png", classifier_sha256="b"*64)
        record.update(changes)
        accepted, receipt = self.root/"accepted.csv", self.root/"receipt.json"
        table(accepted, [record])
        write_json(receipt, dict(complete=True, output_sha256={"accepted.csv": sha(accepted)},
                                folds_sha256=sha(self.repo/"manifests/patient_folds.csv"),
                                effective_policy={"blur_reject_below": 10}))
        (self.root/"variant.png").write_bytes(b"placeholder; hash is checked on image reading")
        specs = [dict(design="resize", namespace="half-A", accepted_csv=str(accepted),
                      receipt=str(receipt), image_root=str(self.root))]
        self.args.synthetic_inputs = self.root/"inputs.json"
        write_json(self.args.synthetic_inputs, specs)

    def run_prepare(self):
        with patch.dict("sys.modules", {"loader": self.fake_loader}):
            prepare(self.args)

    def test_disjoint_patients_and_all_accepted_variant_retained(self):
        self.synthetic()
        self.run_prepare()
        plan = json.loads((self.args.out/"plan.json").read_text())
        self.assertEqual(set(plan["pools"]["real"]), {"T1", "T2"})
        self.assertEqual(set(plan["validation"]), {"V1", "V2"})
        selected = plan["pools"]["resize"]["T1"]
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(r["kind"] == "synthetic" for r in selected), 1)
        self.assertEqual(plan["class_weight"], 1)

    def test_held_out_parent_is_refused(self):
        self.synthetic(patch_id="V1-0", patient_id="V1", assigned_fold="0", slide_id="V1-slide")
        with self.assertRaisesRegex(ValueError, "held out"):
            self.run_prepare()
        self.assertFalse(self.args.out.exists())

    def test_wrong_label_and_source_pixels_and_qc_decision_are_refused(self):
        for change in (dict(mgmt_patient_label="unmethylated"), dict(pixel_sha256="c"*64),
                       dict(pixel_qc_decision="reject"), dict(classifier_path="../escape.png")):
            self.synthetic(**change)
            with self.assertRaises(ValueError):
                self.run_prepare()

    def test_corrupt_qc_and_duplicate_variants_are_refused(self):
        self.synthetic()
        with (self.root/"accepted.csv").open("a") as handle:
            handle.write("tampering")
        with self.assertRaisesRegex(ValueError, "changed accepted"):
            self.run_prepare()
        self.synthetic()
        specifications = json.loads(self.args.synthetic_inputs.read_text())
        write_json(self.args.synthetic_inputs, specifications*2)
        with self.assertRaisesRegex(ValueError, "Duplicate variant"):
            self.run_prepare()

    def test_existing_output_is_preserved(self):
        self.run_prepare()
        original = (self.args.out/"plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_prepare()
        self.assertEqual((self.args.out/"plan.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
