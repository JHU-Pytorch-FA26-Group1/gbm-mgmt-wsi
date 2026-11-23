"""A provisional Macenko reference tile per fold, drawn by a fixed rule until the team decides.

The proposal says the stain reference comes from the training fold; it does not say how the tile
is chosen. This draws a fold's training tiles (TCGA only, CPTAC is always external) in a seeded
random order and takes the first one Macenko can be fitted on. It does not screen for blood, ink
or typical staining.
"""
import json
from pathlib import Path

import numpy as np

import loader
from common import sha256_file

REFERENCE_PIXELS_PER_TILE = 500


def fit_macenko(rgb):
    """Fit a Macenko normalizer on one tile; raises if the tile cannot define a stain."""
    # pylint: disable=import-outside-toplevel
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    normalizer = MacenkoNormalizer()
    normalizer.fit(rgb.copy())   # Macenko may modify its input array
    return normalizer


def choose_references(tiles, reader, seed, folds=range(5), fit=fit_macenko):
    """{fold name: patch_id}: each fold's first fittable tile from its training patients."""
    chosen = {}
    for fold in folds:
        marked = loader.with_roles(tiles, fold)
        candidates = marked[marked["role"] == "train"]
        order = np.random.default_rng([seed, fold]).permutation(len(candidates))
        candidates = candidates.iloc[order]
        for _, row in candidates.iterrows():
            try:
                fit(reader.read(row))
            except Exception:  # pylint: disable=broad-exception-caught
                continue   # Macenko raises various errors on tiles with too little stain
            chosen[str(fold)] = row["patch_id"]
            break
        else:
            raise RuntimeError(f"no tile of fold {fold}'s training patients could be fitted")
    return chosen


def _sampled_rows(rows, per_patient, seed):
    """Up to `per_patient` seeded random rows of each patient, patients in sorted order."""
    for index, (_, mine) in enumerate(rows.groupby("patient_id", sort=True)):
        order = np.random.default_rng([seed, index]).permutation(len(mine))[:per_patient]
        yield from (row for _, row in mine.iloc[order].iterrows())


def _reference_from(stains, od_pixels):
    """(ref_HE 3 x 2, ref_max_C): the median stain matrix and its 99th percentile concentrations."""
    matrix = np.median(np.stack(stains), axis=0)
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    concentrations = np.linalg.lstsq(matrix.T, od_pixels.T, rcond=-1)[0].T
    return matrix.T.astype(np.float32), np.percentile(concentrations, 99, axis=0).astype(np.float32)


def _stain_and_tissue_od(rgb, extractor):
    """(stain matrix, optical density of tissue pixels) of one tile; raises if it has no stain."""
    # pylint: disable=import-outside-toplevel
    from tiatoolbox.utils.misc import get_luminosity_tissue_mask
    from tiatoolbox.utils.transforms import rgb2od
    stain = extractor.get_stain_matrix(rgb.copy())   # TIAToolbox may edit its input
    tissue = get_luminosity_tissue_mask(rgb.copy(), threshold=0.8).ravel()
    return stain, rgb2od(rgb.copy()).reshape(-1, 3)[tissue]


def _collect(rows, reader, per_patient, seed):
    """([stain matrix], [tissue OD sample], tiles that failed) over the sampled tiles of `rows`."""
    from tiatoolbox.tools.stainextract import MacenkoExtractor  # pylint: disable=import-outside-toplevel
    extractor = MacenkoExtractor()
    pixel_rng = np.random.default_rng([seed, 1_000_000])   # apart from the tile draws
    stains, pixels, failed = [], [], 0
    for row in _sampled_rows(rows, per_patient, seed):
        try:
            stain, od = _stain_and_tissue_od(reader.read(row), extractor)
        except Exception:  # pylint: disable=broad-exception-caught
            failed += 1   # Macenko raises various errors on tiles with too little stain
            continue
        stains.append(stain)
        pixels.append(od[pixel_rng.permutation(len(od))[:REFERENCE_PIXELS_PER_TILE]])
    return stains, pixels, failed


def build_cohort_reference(tiles, reader, seed, per_patient=100, sources=("TCGA",)):
    """(ref_HE, ref_max_C, facts): one stain reference for every patient of `sources`.

    Up to `per_patient` seeded random tiles per patient, so every patient weighs the same. Each
    tile's stains come from TIAToolbox's own Macenko extractor, the one that estimates a tile's
    stains when it is normalized. ref_HE (3 x 2, one stain per column) is their median;
    ref_max_C is the 99th percentile of the stain concentrations of tissue pixels under it.
    """
    rows = tiles[tiles["source"].isin(sources)]
    stains, pixels, failed = _collect(rows, reader, per_patient, seed)
    if not stains:
        raise RuntimeError(f"no tile of {list(sources)} could be fitted")
    ref_he, ref_max_c = _reference_from(stains, np.vstack(pixels))
    facts = {"sources": list(sources), "seed": seed, "per_patient": per_patient,
             "patients": rows["patient_id"].nunique(), "tiles_used": len(stains),
             "tiles_failed": failed, "tissue_pixels": sum(len(p) for p in pixels),
             "stain_std_across_tiles": np.stack(stains).std(axis=0).tolist()}
    return ref_he, ref_max_c, facts


def save_cohort_reference(path, ref_he, ref_max_c, facts):
    """Write the reference as `path` (.npz, the arrays) and `path`.json (the arrays and `facts`)."""
    path = Path(path)
    np.savez(path, ref_HE=ref_he, ref_max_C=ref_max_c)
    path.with_suffix(".json").write_text(json.dumps(
        {**facts, "ref_HE": ref_he.tolist(), "ref_max_C": ref_max_c.tolist(),
         "npz_sha256": sha256_file(path)}, indent=2), encoding="utf-8")
