"""Real HDF5/PNG source fixtures and orchestration gates; no GPU inference."""
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from PIL import Image

from alternative_models import half_b_generation as b
from alternative_models import half_b_package as pre
from alternative_models.tests import test_half_b_package as package_fixtures

REPO = Path(__file__).resolve().parents[2]


def templates():
    candidates = [Path(os.environ.get("HIS_B_TEST_TEMPLATES", "/workspace/persist/crivas4-microscopy/gbm-mgmt-wsi/experiments/hismmdm-pilot")),
                  REPO / "experiments/hismmdm-pilot", REPO.parent / "resize-matched-work/experiments/hismmdm-pilot"]
    for path in candidates:
        if all((path / (n + ".py")).is_file() for n in b.NAMES):
            return path
    raise RuntimeError("Set HIS_B_TEST_TEMPLATES to the directory containing all four Half A generation engines")


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Generation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.package = self.base / "packages/B_qc-v5"
        self.package.mkdir(parents=True)
        folds = pre.read_rows(REPO / "manifests/patient_folds.csv")
        labels = {r["patient_id"]: r["mgmt_status"].lower() for r in pre.read_rows(REPO / "manifests/patient_labels.csv")}
        frozen = b.read(REPO / "manifests/collection_manifest.json")
        ownership = {r["patient_id"]: r["half"] for r in package_fixtures.split.build_patient_split(frozen["files"])}
        # Actual canonical IDs/labels/folds, synthetic fixture image bytes.
        selected = []
        for label in ("methylated", "unmethylated"):
            selected += [r for r in folds if ownership[r["patient_id"]] == "B" and r["source"] == "TCGA" and r["fold"] != "0"
                         and labels[r["patient_id"]] == label][:2]
        extra = [next(r for r in folds if ownership[r["patient_id"]] == "B" and r["source"] == "TCGA" and r["fold"] == "0"),
                 next(r for r in folds if ownership[r["patient_id"]] == "B" and r["source"] == "CPTAC")]
        self.fold_rows, self.label_map, self.ownership = folds, labels, ownership
        self.source_people, self.extra = selected, extra
        self.make_package(4)
        refs = b.locations(self.base)["a_original"] / "references"
        refs.mkdir(parents=True)
        shutil.copyfile(REPO / "manifests/patient_labels.csv", refs / "patient_labels.csv")
        self.args = SimpleNamespace(base=self.base, repo=REPO)
        self.original = dict(seed=42, verify_per_class=1)
        self.people = {r["patient_id"] for r in selected}

    def make_package(self, per_patient):
        rows, payloads = [], []
        for patient_index, r in enumerate(self.source_people + self.extra):
            for i in range(per_patient if patient_index < 4 else 1):
                rgb = np.full((224, 224, 3), (30+patient_index*25, 20+(i*30)%200, 120), dtype=np.uint8)
                stream = io.BytesIO()
                Image.fromarray(rgb).save(stream, format="PNG")
                payloads.append(np.frombuffer(stream.getvalue(), dtype=np.uint8))
                rows.append(dict(source=r["source"], patient_id=r["patient_id"], slide_id=f"slide-{patient_index}-{i%2}",
                    fold=r["fold"], patch_id=f"patch-{patient_index}-{i}", role="tissue", x_target=i*224,
                    y_target=patient_index*224, stored="True", h5="tiles/fixture.h5", h5_index=len(rows),
                    pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), decided_by="automatic_qc"))
        path = self.package / "tiles/fixture.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as h5:
            tiles = h5.create_dataset("tiles", (len(payloads),), dtype=h5py.vlen_dtype(np.uint8))
            for i, data in enumerate(payloads):
                tiles[i] = data
        save_csv(self.package / "modeling_patches.csv", rows)
        self.manifest = dict(kind="package", subset="B", complete=True, tiling_version="tiling-v7",
            config_sha256=b.digest(REPO / "configs/tiling.yaml"), folds_sha256=b.digest(REPO / "manifests/patient_folds.csv"),
            policy_version="qc-v5+qc-v5-cptac", code_version=7, policy_sha256="fixture-policy",
            counts=dict(tiles=len(rows)),
            files={n: dict(bytes=(self.package/n).stat().st_size, sha256=b.digest(self.package/n))
                   for n in ("tiles/fixture.h5", "modeling_patches.csv")})
        b.write(self.package / "package_manifest.json", self.manifest)

    def prepared(self, e, name="original"):
        stage = self.base / name
        stage.mkdir()
        rows = b.prepare_sources(self.args, e, self.original, stage, per_class=4)
        return stage, rows

    def weights(self, e):
        root = self.base / "weights"
        root.mkdir(exist_ok=True)
        checkpoint = root / e.core.WEIGHT_NAME
        checkpoint.write_bytes(b"test checkpoint bytes; no model loaded")
        e.core.WEIGHT_SHA256 = b.digest(checkpoint)
        b.write(root / "weights.json", dict(family="HMU-1st", archive_md5=e.core.ARCHIVE_MD5,
            archive_member="/".join(e.core.WEIGHT_MEMBER), filename=e.core.WEIGHT_NAME, sha256=e.core.WEIGHT_SHA256))
        return root

    def test_real_hdf5_preparation_uses_same_sampler_and_excludes_held_out_and_cptac(self):
        with b.engines(templates(), portable=True) as e:
            stage, rows = self.prepared(e)
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["patient_id"] for r in rows}, self.people)
            self.assertEqual(Counter(r["mgmt_patient_label"] for r in rows), dict(methylated=4, unmethylated=4))
            self.assertEqual(set(Counter(r["patient_id"] for r in rows).values()), {2})
            self.assertTrue(all(r["source"] == "TCGA" and r["assigned_fold"] != "0" for r in rows))
            self.assertTrue(all("tissue_fraction" not in r and "laplacian_variance" not in r for r in rows))
            self.assertEqual(Counter(r["mgmt_patient_label"] for r in rows if r["repeat_checks"]), dict(methylated=1, unmethylated=1))
            _, repeated = self.prepared(e, "repeat")
            self.assertEqual(rows, repeated)
            self.assertEqual(b.digest(stage / "references/input_manifest.csv"),
                             b.digest(self.base / "repeat/references/input_manifest.csv"))

    def test_portable_validator_checks_real_reference_pixels_and_weights(self):
        with b.engines(templates(), portable=True) as e:
            stage, rows = self.prepared(e)
            args = SimpleNamespace(refs=stage / "references", fold=0, phase="batch", weights=self.weights(e))
            checkpoint, weights, validated, _, allowed = e.legacy.core.validated_inputs(args)
            self.assertEqual(len(validated), 8)
            self.assertEqual(weights["sha256"], b.digest(checkpoint))
            self.assertTrue(self.people <= set(allowed))
            image = args.refs / rows[0]["image_path"]
            image.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                e.legacy.core.validated_inputs(args)

    def test_coordinated_fold_edit_is_still_rejected_by_canonical_fold_map(self):
        with b.engines(templates(), portable=True) as e:
            stage, _ = self.prepared(e)
            refs = stage / "references"
            a = SimpleNamespace(refs=refs, fold=0, phase="batch", weights=self.weights(e))
            rows = pre.read_rows(refs / "input_manifest.csv")
            source = pre.read_rows(refs / "package_source_rows.csv")
            rows[0]["assigned_fold"] = "0"
            next(r for r in source if r["patch_id"] == rows[0]["patch_id"])["fold"] = "0"
            save_csv(refs / "input_manifest.csv", rows)
            save_csv(refs / "package_source_rows.csv", source)
            prep = b.read(refs / "references.json")
            prep.update(manifest_sha256=b.digest(refs / "input_manifest.csv"), source_rows_sha256=b.digest(refs / "package_source_rows.csv"))
            b.write(refs / "references.json", prep)
            with self.assertRaisesRegex(ValueError, "training lineage"):
                e.legacy.core.validated_inputs(a)

    def test_same_original_pixel_and_label_identity_must_hold(self):
        with b.engines(templates(), portable=True) as e:
            stage, _ = self.prepared(e)
            refs = stage / "references"
            args = SimpleNamespace(refs=refs, fold=0, phase="batch", weights=self.weights(e))
            rows = pre.read_rows(refs / "input_manifest.csv")
            rows[0]["mgmt_patient_label"] = "unmethylated" if rows[0]["mgmt_patient_label"] == "methylated" else "methylated"
            save_csv(refs / "input_manifest.csv", rows)
            prep = b.read(refs / "references.json")
            prep["manifest_sha256"] = b.digest(refs / "input_manifest.csv")
            b.write(refs / "references.json", prep)
            with self.assertRaisesRegex(ValueError, "training lineage"):
                e.legacy.core.validated_inputs(args)

    def test_engine_dependency_adapter_preserves_inference_code_and_imports(self):
        old = {name: sys.modules.get(name) for name in b.NAMES}
        with b.engines(templates(), portable=True) as e:
            code = e.legacy.worker.__code__
            self.assertEqual(e.legacy.core.make_tensor, e.core.make_tensor)
            self.assertEqual(e.matched.core.decode, e.core.decode)
            self.assertEqual(e.legacy.core.release_settings, e.core.release_settings)
            self.assertIs(e.matched.legacy, e.legacy)
            self.assertEqual(e.legacy.worker.__code__, code)
            first = e.legacy.seed_for(42, "ref000001", "variant", 1)
            eighth = e.matched.seed_for(dict(seed=42), "ref000001", 8)
            self.assertNotEqual(first, eighth)
        self.assertEqual({name: sys.modules.get(name) for name in b.NAMES}, old)

    def test_crop_can_reuse_its_distinct_recorded_core_without_changing_resize(self):
        code = self.base / "templates"
        code.mkdir()
        for name in b.NAMES:
            shutil.copyfile(templates() / f"{name}.py", code / f"{name}.py")
        (code / "crop_core.py").write_text((code / "his_mmdm_he_augment.py").read_text() + "\n# distinct recorded crop wrapper fixture\n")
        with b.engines(code, portable=True) as e:
            self.assertNotEqual(b.digest(e.crop.core.__file__), b.digest(e.core.__file__))
            self.assertEqual(Path(e.crop.core.__file__).name, "crop_core.py")
            self.assertEqual(Path(e.legacy.core.__file__).name, "his_mmdm_he_augment.py")
            self.assertNotEqual(e.crop.core.make_tensor, e.legacy.core.make_tensor)
            self.assertEqual(e.crop.core.make_tensor.__code__.co_code, e.legacy.core.make_tensor.__code__.co_code)

    def test_full_2000_source_preparation_preserves_exact_matching_and_detects_tampering(self):
        self.make_package(500)
        # Minimal metadata sidecars and an A comparison manifest, as in a real package.
        for name in ("patient_index.csv", "slide_status.csv", "export_settings.json"):
            (self.package / name).write_text("{}" if name.endswith("json") else "patient_id\n")
            self.manifest["files"][name] = dict(bytes=(self.package/name).stat().st_size, sha256=b.digest(self.package/name))
        b.write(self.package / "package_manifest.json", self.manifest)
        a_package = self.base / "packages/A_qc-v5"
        a_package.mkdir()
        b.write(a_package / "package_manifest.json", dict(self.manifest, subset="A"))
        b.write(self.base / "experiments/alternative-models/half-b-onedrive/receipt.json",
                dict(complete=True, manifest_sha256=b.digest(self.package / "package_manifest.json")))
        paths = b.locations(self.base)
        parent, crop = paths["a_original"], paths["a_crop"]
        refs = parent / "references"
        (refs / "originals").mkdir()
        shutil.copyfile(REPO / "manifests/patient_folds.csv", refs / "patient_folds.csv")
        code = self.base / "gbm-mgmt-wsi/experiments/hismmdm-pilot"
        code.mkdir(parents=True)
        for name in b.NAMES:
            shutil.copyfile(templates() / f"{name}.py", code / f"{name}.py")
        # A distinct crop wrapper fixture, with the same inference functions.
        crop_code = crop / "runai/fixture/code"
        a_rows = []
        with b.engines(code, portable=False) as e:
            for label_index, label in enumerate(e.legacy.LABELS):
                patient = next(r for r in self.fold_rows if self.ownership[r["patient_id"]] == "A"
                               and r["source"] == "TCGA" and r["fold"] != "0"
                               and self.label_map[r["patient_id"]] == label)
                for i in range(1000):
                    ref = f"ref{label_index*1000+i+1:06d}"
                    rel = f"originals/{ref}.png"
                    im = Image.new("RGB", (224, 224), (90, 40, 100))
                    im.save(refs / rel)
                    a_rows.append(dict(reference_id=ref, source="TCGA", patient_id=patient["patient_id"],
                        assigned_fold=patient["fold"], slide_id="A-slide", patch_id="A-"+ref,
                        image_path=rel, source_sha256=b.digest(refs/rel), pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                        mgmt_patient_label=label, source_mpp=.5, repeat_checks=i<8, workload_shard=i%2,
                        qc_defect_flags=""))
            save_csv(refs / "input_manifest.csv", a_rows)
            save_csv(parent / "patient_allocation.csv", [dict(patient_id="fixture", selected_sources=2000)])
            b.write(refs / "references.json", dict(held_out_fold=0, manifest_sha256=b.digest(refs / "input_manifest.csv"),
                    inputs_sha256=dict(folds=b.digest(refs/"patient_folds.csv"), labels=b.digest(refs/"patient_labels.csv"))))
            original = dict(protocol="frozen-hmu1st-he-resize-v1", seed=42, held_out_fold=0, source_count=2000,
                sources_per_class=1000, variants_per_source=2, synthetic_count=4000, shard_count=2,
                eta=.3, verify_per_class=8, references_sha256=b.digest(refs/"references.json"),
                allocation_sha256=b.digest(parent/"patient_allocation.csv"))
            b.write(parent / "batch_plan.json", original)
            e.crop.prepare(SimpleNamespace(parent=parent, out=crop))
        crop_code.mkdir(parents=True)
        (crop_code / "his_mmdm_he_augment.py").write_text((code / "his_mmdm_he_augment.py").read_text() + "\n# crop snapshot\n")
        signatures = dict(wrapper_sha256=b.digest(code/"his_mmdm_he_augment.py"),
                          batch_script_sha256=b.digest(code/"his_mmdm_batch.py"),
                          batch_plan_sha256=b.digest(parent/"batch_plan.json"), runtime={}, model_flags={})
        b.write(parent/"generated/shard-0/batch_run.json", dict(signature=signatures))
        b.write(paths["a_extension"]/"batch_plan.json", dict(core_sha256=signatures["wrapper_sha256"],
                legacy_sha256=signatures["batch_script_sha256"], preparer_sha256=b.digest(code/"his_mmdm_resize_matched.py")))
        b.write(crop/"generated/shard-0/batch_run.json", dict(signature=dict(wrapper_sha256=b.digest(crop_code/"his_mmdm_he_augment.py"),
                runner_sha256=b.digest(code/"his_mmdm_corners.py"), batch_size=8, config={}, runtime={},
                batch_plan_sha256=b.digest(crop/"batch_plan.json"))))
        qc_script = code/"qc_synthetics_light.py"
        qc_script.write_text("# fixture QC script, not executed by preparation\n")
        for shard in (0, 1):
            b.write(code/f"shard-{shard}-light-qc-v4-receipt.json", dict(complete=True, protocol="synthetic-light-v4",
                shard=shard, images=2000, counts=dict(accept=2000), effective_policy=dict(blur_reject_below=10),
                parent_policy_sha256="fixture", tiling_sha256="fixture", runtime={},
                code_sha256={str(qc_script): b.digest(qc_script)}))
        b.prepare(self.args)
        p = b.validate(self.base)
        root = paths["root"]
        original_rows = pre.read_rows(root/"resize-original-4000/references/input_manifest.csv")
        crop_rows = pre.read_rows(root/"corners-16000/references/input_manifest.csv")
        self.assertEqual(len(original_rows), 2000)
        self.assertEqual(len(crop_rows), 8000)
        self.assertEqual(Counter(r["patch_id"] for r in crop_rows), Counter(dict.fromkeys((r["patch_id"] for r in original_rows), 4)))
        self.assertEqual(p["planned_resize"], 16000)
        self.assertNotEqual(p["engine_sha256"]["his_mmdm_he_augment"], p["crop_core_sha256"])
        audit_args = SimpleNamespace(base=self.base, repo=self.base / "audit-repo")
        b.record(audit_args)
        record_path = audit_args.repo / "alternative_models/records/half-B-fold0"
        first = b.digest(record_path / "record_manifest.json")
        b.record(audit_args)
        self.assertEqual(first, b.digest(record_path / "record_manifest.json"))
        self.assertFalse(list(record_path.rglob("*.png")))
        (record_path / "patient_allocation.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Recorded metadata changed"):
            b.record(audit_args)
        with self.assertRaisesRegex(ValueError, "exists"):
            b.prepare(self.args)
        (root/"templates/his_mmdm_batch.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "engine changed"):
            b.validate(self.base)


class Gates(unittest.TestCase):
    def table(self, rows):
        return "Workload Type Framework Status Project Preemptible Pods GPU Alloc.\n" + "\n".join(rows)

    def row(self, name, state="Running", gpu="0.50"):
        return f"{name} Training Runai {state} runai-users Yes 1/1 {gpu}"

    def test_released_design_can_replace_a_while_other_design_continues(self):
        text = self.table([self.row("hismmdm-resize-matched12k-20261121", gpu="0.50"),
                           self.row("someone-elses-model", gpu="1.00")])
        self.assertEqual(b.scheduler_gate(text, "crop")["requested_gpu"], 1.)

    def test_running_pending_and_deleting_same_design_block_submission(self):
        for state in ("Running", "Pending", "Deleting"):
            with self.subTest(state=state):
                text = self.table([self.row("hismmdm-corners-run-20261121", state, "1.00")])
                with self.assertRaisesRegex(ValueError, "not released"):
                    b.scheduler_gate(text, "crop")

    def test_queued_duplicate_and_unrecognized_workload_are_refused(self):
        with self.assertRaisesRegex(ValueError, "already active"):
            b.scheduler_gate(self.table([self.row("hismmdm-halfb-resize-20261124", "Pending", "0.00")]), "resize")
        with self.assertRaisesRegex(ValueError, "Inspect this active"):
            b.scheduler_gate(self.table([self.row("hismmdm-other-pilot")]), "crop")

    def test_missing_a_completion_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "not finalized"):
                b.a_complete(Path(temp), "crop")

    def test_completion_with_live_allocation_still_blocks(self):
        with self.assertRaisesRegex(ValueError, "not released"):
            b.scheduler_gate(self.table([self.row("hismmdm-corners-run-old", "Completed", "1.00")]), "crop")
        self.assertEqual(b.scheduler_gate(self.table([self.row("hismmdm-corners-run-old", "Completed", "0.00")]), "crop")["requested_gpu"], 1.)


if __name__ == "__main__":
    unittest.main()
