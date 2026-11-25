"""Identity/geometry/held-out guards for the exploratory ROI pipeline."""
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

from alternative_models import roi_preview as roi
from alternative_models.roi_checkpoint import sha, verify


def write_csv(path, rows):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def update_manifest(root):
    old = json.loads((root/"package_manifest.json").read_text()) if (root/"package_manifest.json").exists() else {}
    old["files"] = {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.iterdir()
                    if p.is_file() and p.name != "package_manifest.json"}
    (root/"package_manifest.json").write_text(json.dumps(old))


def fixture(root):
    repo = root/"repo"
    (repo/"manifests").mkdir(parents=True)
    folds = []
    for half in ("A","B"):
        for name,source,fold in (("T","TCGA","1"),("V","TCGA","0"),("X","CPTAC","external")):
            folds.append(dict(patient_id=half+name,source=source,fold=fold))
    fold_path = repo/"manifests/patient_folds.csv"
    write_csv(fold_path,folds)
    for half in ("A","B"):
        package = root/f"packages/{half}_qc-v5"
        package.mkdir(parents=True)
        write_csv(package/"patient_index.csv",[dict(patient_id=half+n,model_tiles=25 if n == "T" else 0) for n in ("T","V","X")])
        (package/"export_settings.json").write_text('{}')
        yy,xx = np.indices((1120,1120))
        canvas = np.stack(((xx//8+yy//13)%190+30,(yy//7)%170+50,(xx//9)%160+70),axis=-1).astype(np.uint8)
        rows, blobs = [], []
        for y in range(5):
            for x in range(5):
                i = len(rows)
                tile = canvas[y*224:(y+1)*224,x*224:(x+1)*224]
                stream = io.BytesIO();Image.fromarray(tile).save(stream,format="PNG")
                blobs.append(np.frombuffer(stream.getvalue(),dtype=np.uint8))
                rows.append(dict(patient_id=half+"T",source="TCGA",fold="1",slide_id=half+"S",
                    patch_id=f"{half}{i}",x_target=x*224,y_target=y*224,stored="True",h5="payload.h5",
                    h5_index=i,pixel_sha256=hashlib.sha256(tile.tobytes()).hexdigest()))
        with h5py.File(package/"payload.h5","w") as handle:
            tiles = handle.create_dataset("tiles",shape=(25,),dtype=h5py.vlen_dtype(np.uint8))
            for i,blob in enumerate(blobs):tiles[i]=blob
            handle.create_dataset("patch_id",data=np.array([r["patch_id"] for r in rows],dtype="S"))
            handle.create_dataset("pixel_sha256",data=np.array([r["pixel_sha256"] for r in rows],dtype="S64"))
            handle.attrs.update(patient_id=half+"T",slide_id=half+"S",tile_px=224,target_mpp=0.5)
        write_csv(package/"modeling_patches.csv",rows)
        (package/"package_manifest.json").write_text(json.dumps(dict(kind="package",subset=half,complete=True,
            tiling_version="tiling-v7",folds_sha256=sha(fold_path),config_sha256="fixture",
            code_version="fixture",policy_sha256="fixture")))
        update_manifest(package)
    return repo, canvas


class Runtime:
    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, unused, feed):
        tensor = feed["input"]
        return [np.where(tensor[:,0:1] > tensor[:,1:2],2.,-2.).astype(np.float32)]


class Preview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo,self.canvas = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def select(self):
        return roi.select_rows(self.root,self.repo,0,1,42)[0]

    def test_adjacent_pixels_reconstruct_exactly_without_resizing(self):
        groups = self.select()
        self.assertEqual(set(groups),{"AT","BT"})
        for patient,group in groups.items():
            choices = roi.neighborhoods(group["slides"])
            self.assertEqual(len(choices),1)
            rgb,evidence = roi.read_region(group["root"],group["slides"][choices[0][0]],choices[0])
            np.testing.assert_array_equal(rgb,self.canvas[48:1072,48:1072])
            self.assertEqual(len(evidence),25)

    def test_a_missing_neighbor_cannot_be_filled(self):
        groups = self.select()
        slide = groups["AT"]["slides"]["AS"]
        del slide[448,448]
        self.assertEqual(roi.neighborhoods({"AS":slide}),[])

    def test_duplicate_coordinates_and_held_out_rows_are_refused(self):
        package = self.root/"packages/A_qc-v5"
        rows = roi.read_csv(package/"modeling_patches.csv")
        rows[1]["x_target"] = rows[0]["x_target"]
        write_csv(package/"modeling_patches.csv",rows);update_manifest(package)
        with self.assertRaisesRegex(ValueError,"duplicated"):
            self.select()
        rows[1]["x_target"] = "224"
        rows[1]["fold"] = "0"
        write_csv(package/"modeling_patches.csv",rows);update_manifest(package)
        with self.assertRaisesRegex(ValueError,"Held-out"):
            self.select()

    def test_corrupt_payload_and_wrong_mpp_are_refused(self):
        group = self.select()["AT"]
        choice = roi.neighborhoods(group["slides"])[0]
        path = group["root"]/"payload.h5"
        with h5py.File(path,"r+") as handle:handle.attrs["target_mpp"]=1.0
        with self.assertRaisesRegex(ValueError,"resolution"):
            roi.read_region(group["root"],group["slides"]["AS"],choice)
        with h5py.File(path,"r+") as handle:
            handle.attrs["target_mpp"]=0.5
            rgb = np.full((224,224,3),128,dtype=np.uint8)
            buffer = io.BytesIO();Image.fromarray(rgb).save(buffer,format="PNG")
            handle["tiles"][0] = np.frombuffer(buffer.getvalue(),dtype=np.uint8)
        with self.assertRaisesRegex(ValueError,"Corrupt"):
            roi.read_region(group["root"],group["slides"]["AS"],choice)

    def test_model_coordinates_and_logits_map_back_to_original(self):
        rgb = self.canvas[48:1072,48:1072].copy()
        tensor = roi.inference_tensor(rgb)
        self.assertEqual(tensor[0,1,23,95],rgb[95,23,1])
        self.assertEqual(float(tensor.max()),float(rgb.max()))
        logits = Runtime().run(None,{"input":tensor})[0]
        probability = roi.output_probability(logits)
        np.testing.assert_array_equal(probability>0.5,rgb[...,0]>rgb[...,1])

    def test_metadata_change_and_wrong_checkpoint_are_refused(self):
        with (self.root/"packages/A_qc-v5/modeling_patches.csv").open("a") as handle:handle.write("\n")
        with self.assertRaisesRegex(ValueError,"metadata changed"):
            self.select()
        bad = self.root/"fake.onnx";bad.write_bytes(b'not a model')
        with self.assertRaisesRegex(ValueError,"Checkpoint differs"):
            verify(bad)

    def test_blinded_preview_is_review_only_and_preserves_existing_output(self):
        checkpoint = self.root/"model.onnx";checkpoint.write_bytes(b'fixture only')
        args = SimpleNamespace(base=self.root,repo=self.repo,exclude_fold=0,patients_per_half=1,
            regions_per_patient=1,seed=42,checkpoint=checkpoint,threads=2,out=self.root/"preview")
        with patch.object(roi,"session",return_value=Runtime()):roi.preview(args)
        receipt = json.loads((args.out/"receipt.json").read_text())
        self.assertEqual(receipt["regions"],2)
        self.assertFalse(receipt["production_filter_enabled"])
        self.assertFalse(receipt["valid_for_results"])
        self.assertNotIn("mgmt",(args.out/"review.csv").read_text().lower())
        self.assertNotIn("methylated",(args.out/"review.html").read_text().lower())
        original = (args.out/"receipt.json").read_bytes()
        with self.assertRaisesRegex(ValueError,"New output"):
            roi.preview(args)
        self.assertEqual((args.out/"receipt.json").read_bytes(),original)

    def test_sparse_first_draw_is_replaced_using_geometry_not_predictions(self):
        package = self.root/"packages/A_qc-v5"
        rows = roi.read_csv(package/"modeling_patches.csv")
        original = [dict(row) for row in rows]
        rank = lambda p: hashlib.sha256(f"42|A|{p}".encode()).hexdigest()
        patient = next(f"AN{i}" for i in range(1000) if rank(f"AN{i}") > rank("AT"))
        # Preserve the old sampler's first patient, but remove a central tile.
        rows = [r for r in rows if not (r["x_target"] == "448" and r["y_target"] == "448")]
        for row in original:
            row.update(patient_id=patient,slide_id="ANS",patch_id=patient+row["patch_id"],h5="second.h5")
        rows.extend(original)
        write_csv(package/"modeling_patches.csv",rows)
        index = roi.read_csv(package/"patient_index.csv")
        index.append(dict(patient_id=patient,model_tiles="25"))
        write_csv(package/"patient_index.csv",index)
        fold_path = self.repo/"manifests/patient_folds.csv"
        folds = roi.read_csv(fold_path)
        folds.append(dict(patient_id=patient,source="TCGA",fold="1"))
        write_csv(fold_path,folds)
        for half in ("A","B"):
            root = self.root/f"packages/{half}_qc-v5"
            manifest = json.loads((root/"package_manifest.json").read_text())
            manifest["folds_sha256"] = sha(fold_path)
            (root/"package_manifest.json").write_text(json.dumps(manifest));update_manifest(root)
        shutil.copy2(package/"payload.h5",package/"second.h5")
        with h5py.File(package/"second.h5","r+") as handle:
            handle.attrs.update(patient_id=patient,slide_id="ANS")
            del handle["patch_id"]
            handle.create_dataset("patch_id",data=np.array([r["patch_id"] for r in original],dtype="S"))
        groups,provenance = roi.select_rows(self.root,self.repo,0,1,42)
        self.assertEqual(set(groups),{patient,"BT"})
        status = provenance["region_availability"]["patients"]
        self.assertEqual(status["AT"]["reason"],"insufficient_contiguous_regions")
        self.assertTrue(status[patient]["selected"])
        group = groups[patient];choice = group["plan"][0]
        rgb,_ = roi.read_region(group["root"],group["slides"]["ANS"],choice)
        np.testing.assert_array_equal(rgb,self.canvas[48:1072,48:1072])

    def test_no_feasible_half_writes_availability_without_publishing_images(self):
        package = self.root/"packages/A_qc-v5"
        rows = roi.read_csv(package/"modeling_patches.csv")
        write_csv(package/"modeling_patches.csv",rows[:-1]);update_manifest(package)
        args = SimpleNamespace(base=self.root,repo=self.repo,exclude_fold=0,patients_per_half=4,
            regions_per_patient=1,seed=42,checkpoint=self.root/"not-read.onnx",threads=2,out=self.root/"preview")
        with patch.object(roi,"session") as runtime:
            with self.assertRaisesRegex(ValueError,"A: no training patients"):
                roi.preview(args)
            runtime.assert_not_called()
        self.assertFalse(args.out.exists())
        report = json.loads((self.root/"preview-availability.json").read_text())
        self.assertEqual(report["patients"]["AT"]["complete_5x5_neighborhoods"],0)
        self.assertTrue(report["patients"]["BT"]["selected"])

    def test_selection_is_independent_of_tile_row_order_within_a_slide(self):
        before,_ = roi.select_rows(self.root,self.repo,0,1,42)
        for half in ("A","B"):
            root = self.root/f"packages/{half}_qc-v5"
            rows = roi.read_csv(root/"modeling_patches.csv")
            write_csv(root/"modeling_patches.csv",list(reversed(rows)));update_manifest(root)
        after,_ = roi.select_rows(self.root,self.repo,0,1,42)
        self.assertEqual({p:g["plan"] for p,g in before.items()},{p:g["plan"] for p,g in after.items()})

    def test_spatial_fallback_finds_two_edges_when_a_central_draw_blocks_them(self):
        positions = {(x*224,y*224):{} for y in range(5) for x in range(10)}
        choices = roi.neighborhoods({"S":positions})
        seed = next(i for i in range(100) if sorted(choices,key=lambda c:
            hashlib.sha256(f"{i}|P|{c[:3]}".encode()).hexdigest())[0][1] not in (0,1120))
        selected = roi.spaced_choices(choices,2,seed,"P")
        self.assertEqual(len(selected),2)
        self.assertGreaterEqual(abs(selected[0][1]-selected[1][1]),1120)


if __name__ == "__main__":
    unittest.main()
