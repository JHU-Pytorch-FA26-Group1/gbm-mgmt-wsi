"""Fetch and technically test the published MONAI GBM ONNX checkpoint on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import urllib.request

MODEL_URL = ("https://github.com/hnu-digihealth/monai_gbm_segmentation/releases/"
             "download/v1.0.0-MIE_Release/model_v3.onnx")
MODEL_SHA256 = "d4aef23368e1cfb3d444ecdcefc8d67376e617b19c775c4ed71fe249abfee845"
MODEL_BYTES = 26010685
UPSTREAM_COMMIT = "3c3818233c3c8bcf572c1f8ad6e4181d2fa46fee"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify(path):
    path = Path(path)
    if path.stat().st_size != MODEL_BYTES or sha(path) != MODEL_SHA256:
        raise ValueError("Checkpoint differs from the evaluated release; do not use it")


def download(path):
    path = Path(path)
    if path.exists():
        verify(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial") as temporary:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "GBM-ROI-evaluation"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > MODEL_BYTES:
                    raise ValueError("Unexpectedly large checkpoint response")
                temporary.write(block)
        temporary.flush()
        verify(temporary.name)
        # Exclusive creation protects an existing file from an overlapping invocation.
        with path.open("xb") as output, open(temporary.name, "rb") as source:
            while block := source.read(1024 * 1024):
                output.write(block)


def session(path, threads=2):
    import onnxruntime as ort
    verify(path)
    if not 1 <= threads <= 16:
        raise ValueError("CPU threads must be between 1 and 16")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = runtime.get_inputs(), runtime.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)"
            or inputs[0].shape[1:] != [3, 1024, 1024]
            or outputs[0].shape[1:] != [1, 1024, 1024]):
        raise ValueError("Unexpected ONNX model input/output contract")
    return runtime


def probe(checkpoint, out, threads=2):
    import numpy as np
    import onnx
    import onnxruntime as ort
    if out.exists():
        raise ValueError("Probe output exists; choose a new path")
    verify(checkpoint)
    onnx.checker.check_model(str(checkpoint))
    runtime = session(checkpoint, threads)
    checks = []
    for name, value in (("black", 0.), ("white", 255.)):
        start = time.monotonic()
        result = runtime.run(None, {runtime.get_inputs()[0].name:
            np.full((1, 3, 1024, 1024), value, dtype=np.float32)})[0]
        if result.shape != (1, 1, 1024, 1024) or not np.isfinite(result).all():
            raise ValueError("Nonfinite or incorrect inference output")
        checks.append(dict(input=name, seconds=time.monotonic()-start,
            shape=list(result.shape), logit_min=float(result.min()), logit_max=float(result.max())))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(phase="technical-inference-only", complete=True,
        valid_for_tumor_accuracy=False, production_filter_enabled=False,
        checkpoint_sha256=sha(checkpoint), model_url=MODEL_URL, upstream_commit=UPSTREAM_COMMIT,
        python=platform.python_version(), numpy=np.__version__, onnx=onnx.__version__,
        onnxruntime=ort.__version__, provider=runtime.get_providers(), cpu_threads=threads, checks=checks))
    print(f"CPU inference passed; tumor accuracy has not been evaluated. Receipt: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.download:
        download(args.checkpoint)
    probe(args.checkpoint, args.out, args.threads)


if __name__ == "__main__":
    main()
