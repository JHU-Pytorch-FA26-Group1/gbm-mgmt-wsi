"""Reading slides with TIAToolbox: resolution check, tissue mask, tile grid, tile roles, scores, cap sample.

Used by tile.py (extraction), loader.py (rereading tiles that were not stored) and the notebook, so
all of them use exactly the same mask, grid and score definitions.
"""
import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version as package_version

import cv2
import numpy as np
import openslide
import tiatoolbox
from skimage.filters import threshold_multiotsu, threshold_otsu
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from tiatoolbox.wsicore.wsireader import WSIReader

ROLE_CODES = {"background": 0, "edge": 1, "tissue": 2}
# --masks-only compares these: (label, method, floor). A floor of None means the config's floor.
MASK_VARIANTS = (("two-class", "saturation_otsu", 0),
                 ("three-class", "saturation_multiotsu", 0),
                 ("three-class + floor", "saturation_multiotsu", None))
SCORE_FIELDS = ["mean_r", "mean_g", "mean_b", "gray_median", "mean_saturation", "mean_value",
                "laplacian_variance", "near_white_fraction", "near_black_fraction", "glass_fraction",
                "dark_fraction", "green_fraction"]


def library_versions():
    """Versions of the libraries that shape the outputs, including the native OpenSlide library."""
    import h5py
    versions = {"python": platform.python_version(), "tiatoolbox": tiatoolbox.__version__,
                "openslide": openslide.__library_version__, "openslide_python": openslide.__version__,
                "numpy": np.__version__, "opencv": cv2.__version__, "h5py": h5py.__version__}
    for name in ("pillow", "scikit-image"):
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def open_slide(path, cfg):
    """Open a slide and check its native resolution; return (reader, mpp array, upsampling %).

    Slides coarser than target_mpp are enlarged when read. More than `max_upsample_pct` fails the
    slide instead of silently inventing detail.
    """
    wsi = WSIReader.open(path)
    try:
        mpp = wsi.info.mpp
        if mpp is None or len(mpp) != 2 or not all(np.isfinite(mpp)) or min(mpp) <= 0:
            raise ValueError(f"missing or invalid mpp: {mpp}")
        mpp = np.asarray(mpp, dtype=float)
        upsample = np.maximum(0.0, 100 * (mpp / cfg["target_mpp"] - 1))
        if upsample.max() > cfg["max_upsample_pct"] + 1e-9:
            raise ValueError(f"needs {upsample.max():.2f}% upsampling (mpp {mpp.tolist()})")
    except Exception:
        close_slide(wsi)
        raise
    return wsi, mpp, upsample


def close_slide(wsi):
    """Close the file handles TIAToolbox keeps open (OpenSlide or tifffile)."""
    for name in ("openslide_wsi", "tiff"):
        handle = getattr(wsi, name, None)
        if handle is not None:
            handle.close()


def tissue_mask(thumb, cfg, method=None, floor=None):
    """Return (0/1 uint8 tissue mask, threshold used, Otsu threshold) for an RGB thumbnail.

    The saturation methods threshold the median-blurred HSV saturation channel: tissue is colored,
    background is white or gray. A pixel is tissue if its saturation >= the threshold used, which
    is the Otsu cut raised to at least `floor` (default: `mask_min_saturation`). Small gaps are then
    closed (`mask_close_px`, 0 = off).
    "saturation_multiotsu": the lower cut of a three-class Otsu (background / pale / strong), so pale
    tissue stays tissue on heavily stained slides. On very faint slides that cut can fall inside the
    background noise, which is what the floor is for.
    "saturation_otsu": a two-class Otsu, for comparison. skimage counts pixels ABOVE its threshold as
    foreground, so the cut recorded here is that threshold rounded down, plus 1.
    A thumbnail with too few saturation levels for the method uses the floor alone; its Otsu
    threshold is None.
    "otsu_gray": TIAToolbox's OtsuTissueMasker (grayscale Otsu, no cleanup); both thresholds are None.
    """
    method = method or cfg["mask_method"]
    floor = cfg["mask_min_saturation"] if floor is None else floor
    if method == "otsu_gray":
        return OtsuTissueMasker().fit_transform([thumb])[0].astype(np.uint8), None, None
    saturation = cv2.cvtColor(np.ascontiguousarray(thumb[..., :3]), cv2.COLOR_RGB2HSV)[..., 1]
    saturation = cv2.medianBlur(saturation, cfg["mask_median_px"])
    levels = len(np.unique(saturation))
    otsu = None
    if method == "saturation_multiotsu" and levels >= 3:
        otsu = float(threshold_multiotsu(saturation, classes=3)[0])
    elif method == "saturation_otsu" and levels >= 2:
        otsu = float(math.floor(threshold_otsu(saturation)) + 1)
    threshold = float(max(otsu, floor) if otsu is not None else floor)
    mask = (saturation >= threshold).astype(np.uint8)
    if cfg["mask_close_px"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["mask_close_px"],) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, threshold, otsu


def tile_grid(wsi, cfg):
    """TIAToolbox's full tile grid: a SlidingWindowPatchExtractor with no mask.

    `extractor.coordinate_list` holds every complete `tile_px` window at `target_mpp` as
    [x0, y0, x1, y1] (stride = tile_px, within_bound: no padded tiles at the right/bottom
    margins); its row number is the tile's grid index, and `extractor[i]` reads tile i.
    """
    return SlidingWindowPatchExtractor(
        wsi, cfg["tile_px"], resolution=cfg["target_mpp"], units="mpp", stride=cfg["tile_px"],
        within_bound=True)


def mask_roles(mask, coords, wsi_shape, min_fraction):
    """Classify grid tiles against the thumbnail mask; return (roles, masked fraction).

    Uses the same coordinate scaling and rule as TIAToolbox's PatchExtractor.filter_coordinates,
    so "tissue" here is exactly what an extractor given this mask and min_mask_ratio would keep:
    more than `min_fraction` of the tile is masked, or all of it. "edge" tiles touch the mask
    without being tissue; "background" tiles do not touch it.
    """
    scale = np.array(mask.shape[1::-1]) / np.array(wsi_shape)
    box = coords.copy().astype(np.float32)
    box[:, [0, 2]] *= scale[0]
    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, mask.shape[1])
    box[:, [1, 3]] *= scale[1]
    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, mask.shape[0])
    x0, y0, x1, y1 = box.astype(np.int32).T
    # Masked pixels per box from a summed-area table: one lookup per tile instead of a slice.
    summed = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    summed[1:, 1:] = (mask > 0).cumsum(axis=0).cumsum(axis=1)
    area = np.maximum(x1 - x0, 0).astype(np.int64) * np.maximum(y1 - y0, 0)
    positive = np.where(area > 0, summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0], 0)
    tissue = ((positive == area) | (positive > area * min_fraction)) & (positive > 0) & (area > 0)
    roles = np.where(tissue, ROLE_CODES["tissue"],
                     np.where(positive > 0, ROLE_CODES["edge"], ROLE_CODES["background"]))
    fraction = np.divide(positive, area, out=np.zeros(len(area)), where=area > 0)
    return roles.astype(np.int8), fraction


def tile_scores(img, cfg):
    """Quality scores for an RGB tile (SCORE_FIELDS). Shared names match gbm_patching.py.

    laplacian_variance: variance of the Laplacian of the grayscale tile; low for blurred, empty or
    featureless tiles. mean_saturation / mean_value: HSV means (pale tiles: low saturation, high
    value). glass_fraction: share of pixels with saturation below `glass_max_saturation` and
    brightness above `glass_min_value`. near_white / near_black: share of pixels with every channel
    >= `near_white_value` / <= `near_black_value`. dark_fraction: share with brightness below
    `dark_value` (folds, debris, ink). green_fraction: share with OpenCV hue between `green_hue_min`
    and `green_hue_max` and saturation above `green_min_saturation` (green marker ink).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mean = img.mean(axis=(0, 1))
    return {"mean_r": float(mean[0]), "mean_g": float(mean[1]), "mean_b": float(mean[2]),
            "gray_median": float(np.median(gray)),
            "mean_saturation": float(saturation.mean()), "mean_value": float(value.mean()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "near_white_fraction": float(np.all(img >= cfg["near_white_value"], axis=2).mean()),
            "near_black_fraction": float(np.all(img <= cfg["near_black_value"], axis=2).mean()),
            "glass_fraction": float(((saturation < cfg["glass_max_saturation"])
                                     & (value > cfg["glass_min_value"])).mean()),
            "dark_fraction": float((value < cfg["dark_value"]).mean()),
            "green_fraction": float(((hue >= cfg["green_hue_min"]) & (hue <= cfg["green_hue_max"])
                                     & (saturation > cfg["green_min_saturation"])).mean())}


def stored_tiles(tissue, cfg, slide_id):
    """Which tissue tiles get their pixels stored: all of them, or a seeded random sample.

    Above `max_tiles_per_slide`, a sample of that size is drawn without replacement with a per-slide
    seed (hash of `seed` and the slide ID), so reruns and other machines pick the same tiles. Every
    tissue tile is still read, scored and recorded; tiles not stored can be reread from the slide.
    Returns grid indices in grid order.
    """
    cap = cfg["max_tiles_per_slide"]
    if cap is None or len(tissue) <= cap:
        return tissue
    seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}|{slide_id}".encode()).digest()[:8], "big")
    return np.sort(tissue[np.random.default_rng(seed).choice(len(tissue), size=cap, replace=False)])


def read_tile(wsi, x, y, cfg):
    """Read one tile at its target-resolution top-left (x, y), exactly as the tiler does."""
    img = wsi.read_rect((int(x), int(y)), (cfg["tile_px"],) * 2, cfg["target_mpp"], "mpp",
                        coord_space="resolution")
    return np.ascontiguousarray(img[..., :3])


def version_record():
    """Library versions as a JSON string, for HDF5 attributes."""
    return json.dumps(library_versions())
