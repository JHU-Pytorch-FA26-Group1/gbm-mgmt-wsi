"""Small CPU tensors/mock sampler exercise the actual matched-worker orchestration."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/hismmdm-pilot"))
import his_mmdm_resize_matched as m
import his_mmdm_batch as b
import his_mmdm_he_augment as c


class Model:
    def load_state_dict(self, *a, **kw): pass
    def requires_grad_(self, *a): return self
    def eval(self): return self
    def to(self, *a): return self
    def convert_to_fp16(self): pass
    def __call__(self, x, t, y): return x


class Diffusion:
    num_timesteps = 1000
    timestep_map = list(range(1000))
    def ddim_reverse_sample_loop(self, model, image, **kw):
        assert kw["eta"] == 0 and kw["clip_denoised"] is False
        assert kw["cond_fn"] is None and kw["model_kwargs"]["y"].item() == 10
        assert not torch.is_grad_enabled()
        model(image, torch.tensor([0]), y=kw["model_kwargs"]["y"])
        return image.clone()


def decode(diffusion, model, latent, target, eta, seed, torch_module, device):
    assert not torch.is_grad_enabled() and target.shape == (1,) and target.item() == 10
    assert eta in (0, .3)
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed))
    return (latent + eta*noise*.1).clamp(-1, 1)


class Matched(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.parent, cls.crop, cls.weights = cls.root / "original", cls.root / "crop", cls.root / "weights"
        refs = cls.parent / "references"
        (refs / "originals").mkdir(parents=True)
        cls.weights.mkdir()
        checkpoint = cls.weights / c.WEIGHT_NAME
        checkpoint.write_bytes(b"CPU fixture checkpoint; not a real model")
        cksha = c.digest(checkpoint)
        c.save_json(cls.weights / "weights.json", dict(family="HMU-1st", archive_md5=c.ARCHIVE_MD5,
            filename=c.WEIGHT_NAME, sha256=cksha, archive_member="/".join(c.WEIGHT_MEMBER)))
        rows, folds, labels = [], [], []
        for i in range(4):
            patient = f"TCGA-AA-{i:04d}"
            ref = f"ref{i+1:06d}"
            folds.append(dict(source="TCGA", patient_id=patient, fold="1"))
            labels.append(dict(patient_id=patient, mgmt=int(i < 2)))
            pixels = np.random.default_rng(i).integers(40, 220, (224, 224, 3), dtype=np.uint8)
            image_path = f"originals/{ref}.png"
            Image.fromarray(pixels).save(refs / image_path)
            rows.append(dict(reference_id=ref, source="TCGA", patient_id=patient, assigned_fold="1",
                slide_id=f"slide{i}", patch_id=f"patch{i}", image_path=image_path, source_sha256=c.digest(refs / image_path),
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), source_mpp=.5,
                mgmt_patient_label="methylated" if i < 2 else "unmethylated", workload_shard=i%2,
                repeat_checks=i%2 == 0, morphology="", qc_defect_flags="", tissue_fraction=.9,
                laplacian_variance=150, near_white_fraction=.01, near_black_fraction=0))
        c.save_csv(refs / "patient_folds.csv", folds)
        c.save_csv(refs / "patient_labels.csv", labels)
        c.save_csv(refs / "input_manifest.csv", rows)
        c.save_csv(cls.parent / "patient_allocation.csv", [dict(patient_id=r["patient_id"], selected_sources=1) for r in rows])
        c.save_json(refs / "references.json", dict(held_out_fold=0, batch_mode=True, preview_only=True,
            manifest_sha256=c.digest(refs / "input_manifest.csv"),
            inputs_sha256=dict(folds=c.digest(refs / "patient_folds.csv"), labels=c.digest(refs / "patient_labels.csv"))))
        c.save_json(cls.parent / "batch_plan.json", dict(protocol="frozen-hmu1st-he-resize-v1", held_out_fold=0,
            seed=42, eta=.3, source_count=4, synthetic_count=8, variants_per_source=2, sources_per_class=2,
            shard_count=2, verify_per_class=1, references_sha256=c.digest(refs / "references.json"),
            allocation_sha256=c.digest(cls.parent / "patient_allocation.csv")))
        cls.stack = ExitStack()
        real_device = torch.device
        modules = {"cg_diffusion": ModuleType("cg_diffusion"),
            "cg_diffusion.image_datasets": SimpleNamespace(center_crop_arr=lambda im, size:
                np.asarray(im.resize((size, size), Image.Resampling.BICUBIC))),
            "cg_diffusion.script_util": SimpleNamespace(create_model_and_diffusion=lambda **kw: (Model(), Diffusion()),
                                                        model_and_diffusion_defaults=lambda: {})}
        for mocked in (patch.dict(sys.modules, modules), patch.object(c, "WEIGHT_SHA256", cksha),
            patch.object(c, "release_settings", return_value={}), patch.object(c, "decode", side_effect=decode),
            patch.object(torch, "device", side_effect=lambda *a, **kw: real_device("cpu")),
            patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "set_device"),
            patch.object(torch.cuda, "get_device_name", return_value="CPU TEST DOUBLE"),
            patch.object(torch.cuda, "get_device_capability", return_value=(0, 0)),
            patch.object(torch, "load", return_value={}),
            patch.object(b.subprocess, "run", return_value=SimpleNamespace(stdout="CPU TEST DOUBLE"))):
            cls.stack.enter_context(mocked)
        for shard in (0, 1):
            b.worker(SimpleNamespace(batch=cls.parent, upstream=cls.root, weights=cls.weights, shard=shard, device="cuda:0"))
        crop_rows = []
        for r in c.read_csv(refs / "input_manifest.csv", ("patch_id",)):
            for corner in ("tl", "tr", "bl", "br"):
                crop_rows.append(dict(r, parent_reference_id=r["reference_id"], reference_id=r["reference_id"]+"-"+corner))
        (cls.crop / "references").mkdir(parents=True)
        c.save_csv(cls.crop / "references/input_manifest.csv", crop_rows)
        c.save_json(cls.crop / "batch_plan.json", dict(protocol="frozen-hmu1st-he-corners-v1", held_out_fold=0,
            original_sources=4, crop_inputs=16, synthetic_count=32,
            parent_plan_sha256=c.digest(cls.parent / "batch_plan.json"),
            manifest_sha256=c.digest(cls.crop / "references/input_manifest.csv")))

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        cls.temp.cleanup()

    def prepare(self, name, crop=None):
        out = self.root / name
        m.prepare(SimpleNamespace(parent=self.parent, crop=crop or self.crop, out=out))
        return out

    def args(self, batch, shard):
        return SimpleNamespace(batch=batch, upstream=self.root, weights=self.weights, shard=shard)

    def test_budget_sources_and_seeds(self):
        out = self.prepare("budget")
        p, rows = m.validate(out)
        self.assertEqual((p["existing_outputs"], p["synthetic_count"], p["matched_total"]), (8, 24, 32))
        self.assertEqual((out / "references/input_manifest.csv").read_bytes(),
                         (self.parent / "references/input_manifest.csv").read_bytes())
        _, _, old, _ = m.baseline(self.parent)
        old_seeds = {int(v["seed"]) for rec in old.values() for v in rec["outputs"]}
        new = {m.seed_for(p, r["reference_id"], i) for r in rows for i in m.VARIANTS}
        self.assertEqual(len(new), 24)
        self.assertFalse(new & old_seeds)

    def test_full_six_variant_worker_combines_and_resumes(self):
        out = self.prepare("generate")
        old_images = {p: c.digest(p) for p in self.parent.rglob("*.png")}
        for shard in (0, 1): m.worker(self.args(out, shard))
        m.verify(out)
        combined = c.read_csv(out / "matched_resize_16000.csv", ("variant_id",))
        self.assertEqual(len(combined), 32)
        for ref in {r["reference_id"] for r in combined}:
            self.assertEqual({r["variant_id"] for r in combined if r["reference_id"] == ref},
                             {f"{ref}-{i:02d}" for i in range(1, 9)})
        calls = c.decode.call_count
        m.worker(self.args(out, 0))
        self.assertEqual(c.decode.call_count, calls)
        self.assertEqual({p: c.digest(p) for p in old_images}, old_images)

    def test_wrong_source_pool_is_refused(self):
        wrong = self.root / "wrong-crop"
        shutil.copytree(self.crop, wrong)
        path = wrong / "references/input_manifest.csv"
        rows = c.read_csv(path, ("patch_id",))
        rows[0]["patch_id"] = "NEW-TILE"
        c.save_csv(path, rows)
        p = c.read_json(wrong / "batch_plan.json")
        p["manifest_sha256"] = c.digest(path)
        c.save_json(wrong / "batch_plan.json", p)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            self.prepare("wrong-output", crop=wrong)
        self.assertFalse((self.root / "wrong-output").exists())

    def test_reconstruction_drift_stops_before_variants(self):
        out = self.prepare("drift")
        def wrong_control(*a):
            value = decode(*a)
            return value + .02 if a[4] == 0 else value
        with patch.object(c, "decode", side_effect=wrong_control):
            with self.assertRaisesRegex(ValueError, "Reconstruction differs"):
                m.worker(self.args(out, 0))
        self.assertFalse(list((out / "generated/shard-0/records").glob("*.json")))

    def test_corrupt_saved_output_blocks_resumption(self):
        out = self.prepare("corrupt")
        m.worker(self.args(out, 0))
        path = next((out / "generated/shard-0/variants").glob("*.png"))
        path.write_bytes(b"damaged")
        with self.assertRaises(ValueError): m.worker(self.args(out, 0))

    def test_runtime_drift_stops_before_new_work(self):
        out = self.prepare("runtime")
        with patch.object(torch.cuda, "get_device_name", return_value="DIFFERENT GPU"):
            with self.assertRaisesRegex(ValueError, "runtime/GPU type differs"):
                m.worker(self.args(out, 0))
        self.assertFalse((out / "generated/shard-0").exists())


if __name__ == "__main__": unittest.main()
