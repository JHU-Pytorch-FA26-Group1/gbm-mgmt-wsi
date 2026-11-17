"""A provisional Macenko reference tile per fold, drawn by a fixed rule until the team decides.

The proposal says the stain reference comes from the training fold; it does not say how the tile
is chosen. This draws a fold's training tiles (TCGA only, CPTAC is always external) in a seeded
random order and takes the first one Macenko can be fitted on. It does not screen for blood, ink
or typical staining.
"""
import numpy as np

import loader


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
