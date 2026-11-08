"""Write small synthetic Aperio-style .svs slides for the tests. They are not real tissue.

    python tests/make_slides.py OUTPUT_FOLDER

mixed.svs       0.25 um/px: a strongly stained region, a pale region and a dark patch
faint.svs       0.25 um/px: one very faint region (its three-class Otsu cut needs the floor)
coarse_ok.svs   0.504 um/px: needs 0.8% upsampling, inside the 1% limit
coarse_bad.svs  0.51 um/px: needs 2% upsampling, which tile.py refuses
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile


def _tissue(shape, color, rng, texture=18):
    """Stain color with noise and scattered dark 'nuclei', so tiles have texture."""
    h, w = shape
    noise = rng.normal(0, texture, (h, w, 1))
    nuclei = (rng.random((h, w)) < 0.004).astype(np.float32)
    nuclei = cv2.GaussianBlur(nuclei, (0, 0), 3)[..., None] * 60
    return np.array(color, float)[None, None] + noise - nuclei * np.array([0.6, 1.0, 0.4])


def make(path, mpp, kind, size=(4500, 6000), seed=0):
    """Write one three-level Aperio-style SVS (JPEG tiles, MPP in the description)."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), 242.0) + rng.normal(0, 2, (h, w, 1))      # glass
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "mixed":
        strong = ((xx - 0.3 * w) / (0.22 * w)) ** 2 + ((yy - 0.44 * h) / (0.27 * h)) ** 2 < 1
        pale = ((xx - 0.72 * w) / (0.18 * w)) ** 2 + ((yy - 0.58 * h) / (0.29 * h)) ** 2 < 1
        dark = (abs(xx - 0.3 * w) < 0.042 * w) & (abs(yy - 0.82 * h) < 0.056 * h)
        img[strong] = _tissue((h, w), (190, 110, 165), rng)[strong]
        img[pale] = _tissue((h, w), (236, 214, 230), rng, 6)[pale]
        img[dark] = _tissue((h, w), (45, 30, 45), rng, 5)[dark]
    else:
        pale = ((xx - 0.5 * w) / (0.33 * w)) ** 2 + ((yy - 0.49 * h) / (0.33 * h)) ** 2 < 1
        img[pale] = _tissue((h, w), (238, 226, 234), rng, 4)[pale]
    img = np.clip(img, 0, 255).astype(np.uint8)
    levels = [img, cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA),
              cv2.resize(img, (w // 16, h // 16), interpolation=cv2.INTER_AREA)]
    description = (f"Aperio Image Library v11.2.1\r\n{w}x{h} [0,0 {w}x{h}] (256x256) JPEG/RGB Q=90"
                   f"|AppMag = 40|MPP = {mpp}")
    with tifffile.TiffWriter(path) as tif:
        for i, level in enumerate(levels):
            tif.write(level, tile=(256, 256), compression="jpeg", photometric="rgb",
                      description=description if i == 0 else None, metadata=None)


def make_all(folder):
    """Write the four test slides into `folder`; return {name: path}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    specs = {"mixed": (0.25, "mixed", (4500, 6000), 1), "faint": (0.25, "faint", (4500, 6000), 2),
             "coarse_ok": (0.504, "mixed", (2250, 3000), 3), "coarse_bad": (0.51, "mixed", (2250, 3000), 4)}
    paths = {}
    for name, (mpp, kind, size, seed) in specs.items():
        paths[name] = folder / f"{name}.svs"
        make(paths[name], mpp, kind, size, seed)
    return paths


if __name__ == "__main__":
    print(make_all(sys.argv[1]))
