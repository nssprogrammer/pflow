"""Persistence on MNIST images via gudhi cubical complex.

Convention: we INVERT MNIST so the digit (originally bright on dark
background) becomes dark on a bright background. With sublevel filtration
on the inverted image:
  - H0 features = "ink components"
  - H1 features = HOLES IN THE DIGIT (the loops of 0, 6, 8, 9)

This is the right convention for "kill the holes" to mean "fill in the
loop interiors with bright pixel values."
"""

from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import numpy as np
import gudhi as gd


def invert(img: np.ndarray) -> np.ndarray:
    return 1.0 - img


def compute_persistence_with_locations(
    img: np.ndarray,
    dimensions: Iterable[int] = (0, 1),
    min_persistence: float = 0.0,
) -> List[Dict]:
    """Return list of dicts with birth, death, persistence, dim, and the
    pixel locations of the birth and death critical cells.

    Each dict has keys:
        birth, death, persistence, dim, birth_loc, death_loc.
    Excludes essential (infinite) classes.
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    cc = gd.CubicalComplex(top_dimensional_cells=img)
    # Use -1 (not 0) so gudhi returns ALL pairs including persistence-0 ones.
    # We filter by `min_persistence` ourselves below; this is more reliable
    # across gudhi versions where 0.0 has been interpreted as "drop zero".
    cc.compute_persistence(homology_coeff_field=2, min_persistence=-1)

    cof = cc.cofaces_of_persistence_pairs()
    regular_pairs_by_dim = cof[0]  # exclude essential classes (cof[1])

    flat = img.ravel()
    events: List[Dict] = []
    for dim in dimensions:
        if dim >= len(regular_pairs_by_dim):
            continue
        for pair in regular_pairs_by_dim[dim]:
            b_idx, d_idx = int(pair[0]), int(pair[1])
            b_val, d_val = float(flat[b_idx]), float(flat[d_idx])
            p = d_val - b_val
            if p < min_persistence or p <= 0:
                continue
            events.append({
                'birth': b_val,
                'death': d_val,
                'persistence': p,
                'dim': int(dim),
                'birth_loc': (b_idx // W, b_idx % W),  # (row, col)
                'death_loc': (d_idx // W, d_idx % W),
            })
    return events


def count_betti(img: np.ndarray, threshold: float = 0.5,
                min_h0_area: int = 4, min_h1_area: int = 3) -> Tuple[int, int]:
    """β0, β1 of the binary foreground {img >= threshold}.

    Components smaller than the area floor are treated as noise and not
    counted. Tuned defaults work for 28x28 MNIST: ink components below 4
    pixels and holes below 3 pixels are almost always spurious.
    """
    H, W = img.shape
    fg = img >= threshold
    bg = ~fg

    def label_components(mask):
        labels = np.zeros_like(mask, dtype=np.int32)
        n = 0
        for i0 in range(H):
            for j0 in range(W):
                if not mask[i0, j0] or labels[i0, j0]:
                    continue
                n += 1
                stack = [(i0, j0)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if labels[r, c] or not mask[r, c]:
                        continue
                    labels[r, c] = n
                    stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return labels, n

    fg_labels, n_fg = label_components(fg)
    bg_labels, n_bg = label_components(bg)

    # Foreground components above the area floor.
    fg_areas = np.bincount(fg_labels.ravel())[1:]  # skip label 0 (= bg)
    beta0 = int((fg_areas >= min_h0_area).sum())

    # Background components: identify the outer (border-touching) region and exclude.
    border = set()
    border.update(bg_labels[0, :].tolist())
    border.update(bg_labels[-1, :].tolist())
    border.update(bg_labels[:, 0].tolist())
    border.update(bg_labels[:, -1].tolist())
    border.discard(0)

    n_holes = 0
    for lid in range(1, n_bg + 1):
        if lid in border:
            continue
        if (bg_labels == lid).sum() >= min_h1_area:
            n_holes += 1
    return beta0, n_holes
