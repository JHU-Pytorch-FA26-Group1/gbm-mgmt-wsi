"""Score frozen A+B tile manifests from original WSIs in the quarter-GPU workspace."""
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path("/workspace/persist/crivas4-microscopy")
ALT = BASE / "gbm-mgmt-wsi-alternative-models"
ROOT = BASE / "GBM_MGMT"
EXP = BASE / "experiments/alternative-models"
OUT = EXP / "direct-wsi-roi-v1"
REVIEW = EXP / "roi-preview-fixed-20261125-034609/review"
MODEL = BASE / "gbm-roi-checkpoint-evaluation/checkpoint/model_v3.onnx"
SIZE = 1024
TILE = 224
FIELDS = "half source fold patient_id slide_id patch_id x_target y_target region_x region_y tumor_fraction reason".split()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reader():
    # Reserve the original stdout pipe for binary RGB. Send library logs to stderr.
    binary = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ALT / "data_prep"))
    import slides
    import importlib.util
    pkg = next((BASE / "alternative-models-roi-env/lib").glob("python*/site-packages/torchstain/__init__.py"))
    spec = importlib.util.spec_from_file_location("torchstain", pkg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["torchstain"] = module
    spec.loader.exec_module(module)
    norm = module.normalizers.ReinhardNormalizer(method="modified", backend="numpy")
    ref = np.array(Image.open(REVIEW / "provisional_training_reference.png").convert("RGB"))
    receipt = json.loads((REVIEW / "receipt.json").read_text())
    assert hashlib.sha256(ref.tobytes()).hexdigest() == receipt["provisional_reference_pixel_sha256"]
    norm.fit(ref)
    wsi, _, _ = slides.open_slide(Path(sys.argv[2]), {"target_mpp":0.5, "max_upsample_pct":1.0})
    try:
        width, height = map(int, wsi.slide_dimensions(0.5, "mpp"))
        for bx, by in json.loads(Path(sys.argv[3]).read_text()):
            x = max(0, min(bx*896-64, width-SIZE))
            y = max(0, min(by*896-64, height-SIZE))
            binary.write(f"{x} {y} {width} {height}\n".encode())
            if min(width,height) >= SIZE:
                rgb = slides.read_tile(wsi,x,y,{"tile_px":SIZE,"target_mpp":0.5})
                normalized = norm.normalize(I=rgb)
                if normalized.shape != (SIZE,SIZE,3) or normalized.dtype != np.uint8:
                    raise ValueError("Unexpected normalized RGB")
                binary.write(normalized.tobytes())
        binary.flush()
    finally:
        slides.close_slide(wsi)


def main():
    import torch
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0,str(ALT))
    from alternative_models.roi_preview import inference_tensor, output_probability
    OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/"run.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt = json.loads((REVIEW/"receipt.json").read_text())
    assert digest(MODEL) == receipt["checkpoint_sha256"]
    for name, expected in receipt["source_sha256"].items():
        assert digest(ALT/"alternative_models"/name) == expected
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL),sess_options=opts,providers=["CUDAExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    started = time.monotonic()
    done = previous = tiles = missing = 0
    plan_path = OUT/"current-blocks.json"
    for half in "AB":
        path = BASE/f"packages/{half}_qc-v5/modeling_patches.csv"
        assert digest(path) == receipt["source_metadata"]["packages"][half]["modeling_sha256"]
        with path.open(newline="") as handle:
            for slide, group in itertools.groupby(csv.DictReader(handle),key=lambda r:r["slide_id"]):
                rows = list(group)
                dest = OUT/f"{half}-{slide}.csv"
                blocks = {}
                for r in rows:
                    key = (int(r["x_target"])//896,int(r["y_target"])//896)
                    blocks.setdefault(key,[]).append(r)
                tiles += len(rows)
                if dest.exists():
                    with dest.open(newline="") as saved:
                        old = list(csv.DictReader(saved))
                    assert {r["patch_id"] for r in old} == {r["patch_id"] for r in rows} and len(old)==len(rows)
                    missing += sum(bool(r["reason"]) for r in old)
                    previous += len(blocks)
                    continue
                plan_path.write_text(json.dumps(list(blocks)))
                slide_path = ROOT/rows[0]["slide_relpath"]
                assert slide_path.is_file(), slide_path
                child_env = os.environ.copy()
                for k in ("PYTHONPATH","LD_PRELOAD","LD_LIBRARY_PATH","PYTHONHOME"):
                    child_env.pop(k,None)
                proc = subprocess.Popen([str(BASE/"patch-env/bin/python"),__file__,"read",str(slide_path),str(plan_path)],
                                        stdout=subprocess.PIPE,env=child_env)
                tmp = dest.with_suffix(".partial")
                try:
                    with tmp.open("w",newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(FIELDS)
                        for members in blocks.values():
                            header = proc.stdout.readline()
                            if not header:
                                raise RuntimeError(f"Reader stopped on {slide}; see log above")
                            x,y,w,h = map(int,header.split())
                            mask = None
                            if min(w,h)>=SIZE:
                                data = proc.stdout.read(SIZE*SIZE*3)
                                rgb = np.frombuffer(data,np.uint8).reshape(SIZE,SIZE,3)
                                logits = session.run(None,{session.get_inputs()[0].name:inference_tensor(rgb)})[0]
                                mask = output_probability(logits)>0.5
                            for r in members:
                                tx,ty = int(r["x_target"]),int(r["y_target"])
                                dx,dy = tx-x,ty-y
                                valid = mask is not None and 0<=dx<=SIZE-TILE and 0<=dy<=SIZE-TILE
                                fraction = float(mask[dy:dy+TILE,dx:dx+TILE].mean()) if valid else ""
                                reason = "" if valid else "insufficient_in_bounds_context"
                                missing += int(not valid)
                                writer.writerow([half,*[r[k] for k in FIELDS[1:8]],x,y,fraction,reason])
                            done += 1
                            if done%250==0:
                                rate = done/(time.monotonic()-started)
                                print(f"{previous+done}/323762 blocks; {rate:.2f}/s; remaining estimate {(323762-previous-done)/rate/3600:.1f} h",flush=True)
                    if proc.wait()!=0:
                        raise RuntimeError(f"Reader failed on {slide}")
                    tmp.replace(dest)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
    assert tiles==1987217 and previous+done==323762
    print(f"COMPLETE: {tiles} tiles, {missing} unscored. Results: {OUT}",flush=True)


if __name__=="__main__":
    if sys.argv[1:2]==["read"]:
        reader()
    else:
        main()
