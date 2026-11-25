"""CPU integration: real autograd/Adam/cache files, tiny backbone and deterministic pixels.

These are behavior tests, not GPU benchmarks or scientific performance results.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from alternative_models import engine
from alternative_models.prepare import sha, write_json
from training.models import BagModel

REPO = Path(__file__).resolve().parents[2]


def tiny(backbone, seed, pretrained):
    del backbone, pretrained
    torch.manual_seed(seed)
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
            self.stages = nn.ModuleList([nn.Linear(4, 4)])
            self.head = nn.Identity()
        def forward(self, images):
            return self.head(self.stages[-1](self.stem(images)))
    return BagModel(TinyEncoder(), 4,
                    {"attention_hidden": 3, "attention_dropout": 0.0}), {}


class FakePixels:
    def __init__(self, *args):
        pass

    def tile(self, row, augment_seed=None):
        del augment_seed
        value = (sum(map(ord, row["tile_id"])) % 9 + 1)/10
        return torch.full((3, 2, 2), value)

    def close(self):
        pass


class Engine(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root/"package"
        self.package.mkdir()
        (self.package/"modeling_patches.csv").write_text("fixture; fake pixels are supplied in these tests\n")
        training, validation = {}, {}
        for patient, label, destination in (("T1", 1, training), ("T2", 0, training),
                                            ("V1", 1, validation), ("V2", 0, validation)):
            destination[patient] = [dict(tile_id=f"{patient}-{i}", patient_id=patient,
                kind="synthetic" if i in (0, 1) and patient.startswith("T") else "real",
                parent_patch_id=f"{patient}-parent-{i//2}", slide_id=f"{patient}-slide", label=label,
                x_target=i*224, y_target=0) for i in range(10)]
        self.plan = dict(protocol="alternative-attention-v1", repo=str(REPO),
                         package=str(self.package), modeling_sha256=sha(self.package/"modeling_patches.csv"),
                         folds_sha256=sha(REPO/"manifests/patient_folds.csv"),
                         labels_sha256=sha(REPO/"manifests/patient_labels.csv"),
                         synthetic_inputs=[], seed=42, fold=0, class_weight=1,
                         pools={"resize": training}, validation=validation)
        self.plan_path = self.root/"plan.json"
        write_json(self.plan_path, self.plan)
        self.cache = self.root/"cache"
        fake_loader = SimpleNamespace(TileReader=lambda _: SimpleNamespace(close=lambda: None),
            fit_fold_normalizer=lambda *a: (None, {"stain_matrix_target": [[1,0,0],[0,1,0]], "maxC_target": [[1,1]]}))
        args = SimpleNamespace(plan=self.plan_path, backbone="convnext_v2_t", out=self.cache,
                               reference_patch_id="T1-0", device="cpu", chunk=4)
        with patch.dict("sys.modules", {"loader": fake_loader}), patch.object(engine, "build", tiny), patch.object(engine, "Pixels", FakePixels):
            engine.cache(args)
        self.args = SimpleNamespace(plan=self.plan_path, cache=self.cache, arm="resize",
                                    out=self.root/"run", device="cpu", chunk=3, bag_size=4,
                                    frozen_lr=.01, finetune_lr=.01, frozen_cycles=1,
                                    finetune_cycles=1, patience=2)

    def train(self):
        with patch.object(engine, "build", tiny), patch.object(engine, "Pixels", FakePixels), patch.object(engine, "normalizer_from", lambda _: None):
            engine.train(self.args)

    def test_cache_values_match_initial_encoder_and_ids(self):
        features = engine.Features(self.cache, self.plan_path, self.plan)
        model, _ = tiny("", 42, False)
        rows = self.plan["pools"]["resize"]["T1"]
        expected = model.encode(torch.stack([FakePixels().tile(r) for r in rows])).detach()
        self.assertTrue(torch.allclose(features.get("T1", rows), expected))
        reverse = features.get("T1", list(reversed(rows)))
        self.assertTrue(torch.equal(reverse, features.get("T1", rows).flip(0)))

    def test_two_stages_freeze_then_update_and_cover_every_tile(self):
        self.train()
        initial = torch.load(self.cache/"initial_model.pt", weights_only=True)
        frozen = torch.load(self.args.out/"frozen_best.pt", weights_only=True)
        finetuned = torch.load(self.args.out/"finetune_best.pt", weights_only=True)
        encoder = [k for k in initial if k.startswith("backbone.")]
        self.assertTrue(all(torch.equal(initial[k], frozen[k]) for k in encoder))
        self.assertTrue(any(not torch.equal(frozen[k], finetuned[k]) for k in encoder))
        from alternative_models.prepare import read_csv
        audit = read_csv(self.args.out/"chunk_audit.csv")
        for patient, rows in self.plan["pools"]["resize"].items():
            ids = [tile for row in audit if row["patient_id"] == patient for tile in json.loads(row["tile_ids"])]
            self.assertCountEqual(ids, [r["tile_id"] for r in rows])
            self.assertEqual(len(ids), len(set(ids)))
        record = json.loads((self.args.out/"run_record.json").read_text())
        self.assertTrue(record["complete"] and record["valid_for_results"])
        predictions = read_csv(self.args.out/"predictions.csv")
        self.assertEqual({r["patient_id"] for r in predictions}, {"V1", "V2"})
        self.assertTrue(all(0 <= float(r["score"]) <= 1 for r in predictions))

    def test_one_optimizer_update_per_patient_in_both_stages(self):
        actual, steps = torch.optim.Adam, []
        class CountedAdam(actual):
            def step(self, *a, **kw):
                steps.append(1)
                return super().step(*a, **kw)
        with patch.object(torch.optim, "Adam", CountedAdam):
            self.train()
        # Two patients * two stages, although each fine-tuning patient has three bags.
        self.assertEqual(len(steps), 4)

    def test_stale_initial_weights_and_plan_and_features_are_refused(self):
        features = engine.Features(self.cache, self.plan_path, self.plan)
        record = features.metadata["patients"]["T1"]
        path = self.cache/record["file"]
        saved = path.read_bytes()
        path.write_bytes(saved+b"tampering")
        with self.assertRaisesRegex(ValueError, "Corrupt feature"):
            features.get("T1", self.plan["pools"]["resize"]["T1"])
        path.write_bytes(saved)
        self.plan_path.write_text(self.plan_path.read_text()+" ")
        with self.assertRaisesRegex(ValueError, "stale cache"):
            engine.Features(self.cache, self.plan_path, self.plan)

    def test_failure_leaves_run_incomplete(self):
        with patch.object(engine, "predict", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.train()
        record = json.loads((self.args.out/"run_record.json").read_text())
        self.assertFalse(record["complete"] or record["valid_for_results"])

    def test_early_stopping_restores_best_stage_checkpoint(self):
        self.args.frozen_cycles = self.args.finetune_cycles = 5
        self.args.patience = 1
        losses = iter([.5, .8, .4, .9, .4])
        snapshots = []
        def scripted(model, plan, features, pixels, args, frozen):
            snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            return next(losses), .5, [dict(patient_id="V1", label=1, logit=0, score=.5, fold=0),
                                     dict(patient_id="V2", label=0, logit=0, score=.5, fold=0)]
        with patch.object(engine, "predict", scripted):
            self.train()
        record = json.loads((self.args.out/"run_record.json").read_text())
        self.assertEqual(record["best_cycles"], {"frozen": 1, "finetune": 1})
        selected = torch.load(self.args.out/"finetune_best.pt", weights_only=True)
        self.assertTrue(all(torch.equal(value, snapshots[2][key]) for key, value in selected.items()))
        self.assertTrue(all(torch.equal(value, snapshots[4][key]) for key, value in selected.items()))

    def test_worse_finetuning_retains_frozen_checkpoint_and_smoke_is_not_a_result(self):
        self.args.smoke = True
        losses = iter([.3, .8, .3])
        def scripted(*args, **kwargs):
            return next(losses), .5, [dict(patient_id="V1", label=1, logit=0, score=.5, fold=0),
                                     dict(patient_id="V2", label=0, logit=0, score=.5, fold=0)]
        with patch.object(engine, "predict", scripted):
            self.train()
        record = json.loads((self.args.out/"run_record.json").read_text())
        self.assertTrue(record["complete"])
        self.assertFalse(record["valid_for_results"])
        self.assertEqual(record["selected_stage"], "frozen")
        selected = torch.load(self.args.out/"selected_best.pt", weights_only=True)
        frozen = torch.load(self.args.out/"frozen_best.pt", weights_only=True)
        self.assertTrue(all(torch.equal(value, frozen[key]) for key, value in selected.items()))
        metrics = json.loads((self.args.out/"patient_metrics.json").read_text())
        self.assertEqual(metrics["confusion_matrix"], [[0,1],[0,1]])

    def test_budget_above_200_is_refused_before_creating_run(self):
        self.args.frozen_cycles = 20
        self.args.finetune_cycles = 181
        with self.assertRaisesRegex(ValueError, "200"):
            self.train()
        self.assertFalse(self.args.out.exists())


if __name__ == "__main__":
    unittest.main()
