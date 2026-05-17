"""Forward process for PFlow-T: kill H1 features in ascending order of persistence.

DESIGN NOTE (changed in v3): The earlier implementation used gudhi's
death-cell locations as fill centers. For sublevel-set cubical persistence
on inverted MNIST images, gudhi reports the saddle pixel where the loop
CLOSES topologically — this sits on the ring stroke, not inside the loop.
Filling a disk around the death cell therefore added ink on top of the
existing stroke and never actually filled the holes (β1 stayed constant
across the entire t sweep — see sweep.png from the v2 smoke test).

We now detect loop interiors directly via connected-components on the
background. Each detected interior is one H1 feature; we fill them in
ascending order of area (a robust proxy for persistence — small loops
have smaller persistence in the cubical filtration anyway).

The `schedule` argument from the previous API is kept for backward
compatibility but is now ignored — we recompute interiors from the
input image each call. This costs ~1 ms per call for 28x28; not a
bottleneck during training.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np


# =========================================================================
# Connected-components on a boolean mask (pure numpy, no scipy required)
# =========================================================================

def _flood_fill_components(mask: np.ndarray) -> tuple:
    """Label 4-connected components of a boolean mask.

    Returns (labels, n_components) where labels[i,j] is 0 for False
    cells and a positive integer in [1, n_components] for True cells.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
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
                stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return labels, n


# =========================================================================
# Loop-interior detection
# =========================================================================

def find_loop_interiors(image: np.ndarray, threshold: float = 0.5) -> List[Dict]:
    """Find loop interiors (digit holes) in a 2D image.

    A loop interior is a connected component of the BACKGROUND (pixels
    with intensity < threshold) that does NOT touch the image border.

    Returns a list of dicts {mask, area} sorted by ASCENDING area
    (smallest first — these are killed first in the forward process).
    """
    H, W = image.shape
    bg = image < threshold
    labels, n = _flood_fill_components(bg)

    # Components touching the border are the outer background, not interiors.
    border = set()
    border.update(labels[0, :].tolist())
    border.update(labels[-1, :].tolist())
    border.update(labels[:, 0].tolist())
    border.update(labels[:, -1].tolist())
    border.discard(0)

    interiors = []
    for lid in range(1, n + 1):
        if lid in border:
            continue
        mask = (labels == lid)
        area = int(mask.sum())
        if area < 2:
            continue  # singleton noise
        interiors.append({'mask': mask, 'area': area})

    interiors.sort(key=lambda h: h['area'])
    return interiors


# =========================================================================
# Forward process
# =========================================================================

def persistence_melt(
    image: np.ndarray,
    t: float,
    schedule: Optional[List[Dict]] = None,
    fill_radius: float = 2.5,
    return_schedule: bool = False,
    threshold: float = 0.5,
):
    """Apply the forward process at time t in [0, 1].

    Args:
        image: (H, W) float array in [0, 1] — original MNIST digit.
        t: time in [0, 1]. t=0 -> unchanged; t=1 -> all loops fully filled.
        schedule: ignored (legacy API; see module docstring).
        fill_radius: ignored.
        return_schedule: if True, also return the interiors list.
        threshold: ink/background threshold used for hole detection.

    Returns:
        melted image of same shape, values in [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)
    interiors = find_loop_interiors(image, threshold=threshold)
    if not interiors:
        out = image.copy()
        return (out, interiors) if return_schedule else out

    total_area = sum(h['area'] for h in interiors)
    target_area = float(np.clip(t, 0.0, 1.0)) * total_area

    out = image.copy()
    acc = 0.0
    for h in interiors:
        if acc >= target_area:
            break
        remaining = target_area - acc
        if remaining >= h['area']:
            amp = 1.0
            acc += h['area']
        else:
            amp = float(remaining / h['area'])
            acc = target_area
        # Fill: lift background pixels in the loop interior up to amp * 1.0
        # (foreground intensity), using max so we don't dim existing pixels.
        out = np.where(h['mask'], np.maximum(out, amp), out)

    return (out, interiors) if return_schedule else out


# =========================================================================
# Legacy compat: dataset.py imports build_h1_kill_schedule but the new
# forward process doesn't use it. We keep the function for API stability.
# =========================================================================

def build_h1_kill_schedule(img_inverted: np.ndarray,
                           min_persistence: float = 0.05) -> List[Dict]:
    """Legacy: still computes the gudhi H1 schedule for callers that want
    persistence values. The forward process no longer uses these directly
    — see module docstring.
    """
    from .persistence import compute_persistence_with_locations
    events = compute_persistence_with_locations(img_inverted, dimensions=(1,))
    events = [e for e in events if e['persistence'] >= min_persistence]
    events.sort(key=lambda e: e['persistence'])
    return events
