"""Persistence landscape vectorization for the stronger baseline.

A persistence landscape (Bubenik 2015) of a persistence diagram is a
sequence of piecewise-linear functions:

  λ_k(t) = k-th largest of { f_(b,d)(t) : (b, d) in PD }

where f_(b,d)(t) is the tent function:
  f_(b,d)(t) = max(0, min(t - b, d - t)).

We sample λ_1, ..., λ_L on a regular grid of S points and flatten to
get an (L * S)-dimensional descriptor. Both H_0 and H_1 landscapes
are concatenated, giving a 2 * L * S vector.

Reference: Bubenik 2015, "Statistical topological data analysis using
persistence landscapes." Journal of Machine Learning Research 16.
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np


def _tent_values(birth: float, death: float, ts: np.ndarray) -> np.ndarray:
    """Tent function f_(b,d)(t) = max(0, min(t-b, d-t)) evaluated at ts."""
    if death <= birth:
        return np.zeros_like(ts)
    return np.maximum(0.0, np.minimum(ts - birth, death - ts))


def landscape_for_dim(
    events: List[Dict], dim: int, L: int, S: int,
    t_min: float = 0.0, t_max: float = 1.0,
) -> np.ndarray:
    """Compute the L-level persistence landscape on S grid points
    for events of a given homological dimension.

    Returns: (L, S) array. Out-of-range levels are zero-filled.
    """
    ts = np.linspace(t_min, t_max, S, dtype=np.float32)
    relevant = [(e['birth'], e['death']) for e in events
                if e['dim'] == dim and e['death'] > e['birth']]
    if not relevant:
        return np.zeros((L, S), dtype=np.float32)

    # Evaluate every tent at every grid point: (n_features, S) matrix.
    tents = np.stack([_tent_values(b, d, ts) for b, d in relevant], axis=0)
    # For each grid point, sort tent values descending; take the top L.
    sorted_tents = -np.sort(-tents, axis=0)  # (n_features, S), descending
    out = np.zeros((L, S), dtype=np.float32)
    n_avail = sorted_tents.shape[0]
    out[:min(L, n_avail)] = sorted_tents[:min(L, n_avail)]
    return out


def landscape_descriptor(
    events: List[Dict], L: int = 4, S: int = 8,
    t_min: float = 0.0, t_max: float = 1.0,
) -> np.ndarray:
    """Full 2 * L * S descriptor stacking H0 and H1 landscapes.

    Defaults (L=4, S=8) give a 64-dim vector — comparable to what TAGG
    and TopoDiffusionNet use, and a strict superset of information vs.
    the 5-d summary.
    """
    h0 = landscape_for_dim(events, dim=0, L=L, S=S, t_min=t_min, t_max=t_max)
    h1 = landscape_for_dim(events, dim=1, L=L, S=S, t_min=t_min, t_max=t_max)
    return np.concatenate([h0.flatten(), h1.flatten()]).astype(np.float32)
