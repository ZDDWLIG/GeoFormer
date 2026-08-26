"""First-break picking — NPP (Nearest-Point Picking)."""

from __future__ import annotations

import numpy as np

def _npp_transition_mask(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    """Detect 0->1 transitions along time axis (axis=0)."""
    b = prob_map > threshold
    prev_high = np.zeros_like(b, dtype=bool)
    prev_high[1:, :] = b[:-1, :]
    return b & ~prev_high

def pick_npp_from_prob_map(
    prob_map: np.ndarray,
    *,
    threshold: float = 0.5,
    npp_anchor: str = "earliest",
) -> np.ndarray:
    """Nearest-Point Picking: prob_map (nt, nx) -> (nx,) first-break sample indices."""
    prob_map = np.asarray(prob_map, dtype=np.float32)
    if prob_map.ndim != 2:
        raise ValueError(f"NPP requires prob_map to be 2D (nt, nx), got {prob_map.shape}")
    nt, nx = prob_map.shape
    edges = _npp_transition_mask(prob_map, threshold)
    picks = np.zeros(nx, dtype=np.float32)

    c0 = np.flatnonzero(edges[:, 0])
    col0 = prob_map[:, 0]
    if c0.size == 0:
        picks[0] = float(np.argmax(col0))
    elif npp_anchor == "earliest":
        picks[0] = float(c0[0])
    elif npp_anchor == "max_grad":
        g0 = np.empty(nt, dtype=np.float32)
        g0[0] = col0[0]
        g0[1:] = col0[1:] - col0[:-1]
        picks[0] = float(c0[np.argmax(g0[c0])])
    else:
        raise ValueError("npp_anchor must be 'earliest' or 'max_grad'")

    prev = picks[0]
    for j in range(1, nx):
        cj = np.flatnonzero(edges[:, j])
        if cj.size == 0:
            picks[j] = prev
            continue
        k = int(np.argmin(np.abs(cj.astype(np.float32) - prev)))
        picks[j] = float(cj[k])
        prev = picks[j]

    return picks

def extract_picks_from_prediction(
    prob_map,
    threshold=0.5,
    dt=1.0,
    npp_anchor: str = "earliest",
):
    """NPP first-break picking.  prob_map shape: (time, traces)."""
    prob_map = np.asarray(prob_map, dtype=np.float32)
    if prob_map.ndim != 2:
        raise ValueError(f"prob_map should be 2D, got {prob_map.shape}")
    picks = pick_npp_from_prob_map(prob_map, threshold=threshold, npp_anchor=npp_anchor)
    return picks * dt
