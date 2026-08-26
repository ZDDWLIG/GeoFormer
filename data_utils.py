"""SEGY I/O, normalization, coordinate bounds — extracted from dataset/utils.py.

Self-contained; only depends on numpy, scipy, and segyio.
"""

from __future__ import annotations

import shutil
import os

import numpy as np
import segyio

def read_segy_data(segy_path):
    """Load traces, per-trace 4D headers, sample count, and bin interval (seconds if >0)."""
    with segyio.open(segy_path, "r", strict=False, ignore_geometry=True) as f:
        data = f.trace.raw[:]
        ns = f.bin[segyio.BinField.Samples]
        interval_us = int(f.bin[segyio.BinField.Interval])
        coords = []
        for i in range(f.tracecount):
            h = f.header[i]
            coords.append([h[73], h[77], h[81], h[85]])
        coords = np.array(coords, dtype=np.float32)
    sample_interval_sec = (interval_us * 1e-6) if interval_us > 0 else None
    return data, coords, ns, sample_interval_sec

def _segy_header_scalar_to_factor(scalar: int) -> float:
    """SEGY Rev1: 0 as 1; positive multiply; negative divide by |scalar|."""
    if scalar == 0:
        return 1.0
    if scalar < 0:
        return 1.0 / float(-scalar)
    return float(scalar)

def read_segy_geom_per_trace(segy_path: str) -> np.ndarray:
    """Per-trace (offset, rel_elev) in physical units."""
    tf = segyio.TraceField
    with segyio.open(segy_path, "r", strict=False, ignore_geometry=True) as f:
        n = f.tracecount
        out = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            h = f.header[i]
            cf = _segy_header_scalar_to_factor(int(h[tf.SourceGroupScalar]))
            hf = _segy_header_scalar_to_factor(int(h[tf.ElevationScalar]))
            sx = int(h[tf.SourceX]) * cf
            sy = int(h[tf.SourceY]) * cf
            gx = int(h[tf.GroupX]) * cf
            gy = int(h[tf.GroupY]) * cf
            out[i, 0] = float(np.hypot(sx - gx, sy - gy))
            out[i, 1] = (
                int(h[tf.SourceSurfaceElevation]) * hf
                - int(h[tf.ReceiverGroupElevation]) * hf
            )
    return out

def _apply_coord_bounds(coords_raw: np.ndarray, bounds) -> np.ndarray:
    """Min-max to [0,1] per dimension; degenerate dims → 0.5."""
    lo, hi = bounds
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    assert lo.shape == hi.shape == (coords_raw.shape[1],), (
        f"bounds shape mismatch with coord dims: lo{lo.shape} hi{hi.shape} coords{coords_raw.shape}"
    )
    out = coords_raw.astype(np.float64).copy()
    for dim in range(out.shape[1]):
        d = hi[dim] - lo[dim]
        if d > 1e-8:
            out[:, dim] = (out[:, dim] - lo[dim]) / d
        else:
            out[:, dim] = 0.5
    return out

def _chunked_percentile(data: np.ndarray, q: float, chunk_traces: int = 50000) -> float:
    """Approximate the q-th percentile of |data| without a full-sized abs copy."""
    n_traces = data.shape[0]
    chunk_size = max(1, min(chunk_traces, n_traces))
    samples = []
    for start in range(0, n_traces, chunk_size):
        end = min(start + chunk_size, n_traces)
        chunk = np.abs(data[start:end])
        k = max(1, min(chunk.size // 100, 500_000))
        idx = np.linspace(0, chunk.size - 1, k, dtype=np.int64)
        samples.append(np.partition(chunk.ravel(), k - 1)[idx])
    all_samples = np.concatenate(samples)
    return float(np.percentile(all_samples, q))

def _chunked_trace_percentile(data: np.ndarray, q: float, chunk_traces: int = 50000) -> np.ndarray:
    """Per-trace percentile of |data|, chunked to bound peak memory."""
    n_traces, ns = data.shape
    chunk_size = max(1, min(chunk_traces, n_traces))
    result = np.empty(n_traces, dtype=np.float64)
    for start in range(0, n_traces, chunk_size):
        end = min(start + chunk_size, n_traces)
        chunk = data[start:end].copy()
        abs_chunk = np.abs(chunk, out=chunk)
        result[start:end] = np.percentile(abs_chunk, q, axis=1)
    return result

def _normalize_seismic(
    seismic_data: np.ndarray,
    clip_percentile: float,
    mode: str = "file",
    verbose: bool = True,
    _hdr: str = "",
) -> np.ndarray:
    """Clip and normalize seismic data.

    Uses in-place operations where possible to avoid duplicating large arrays.
    """
    x = np.nan_to_num(seismic_data, nan=0.0, copy=True)
    n_traces, ns = x.shape

    if mode == "file":
        clip_value = _chunked_percentile(x, clip_percentile)
        if verbose:
            print(f"{_hdr}\nClip value (per-file, |x| {clip_percentile}% percentile): {clip_value:.4f}")
        scale = clip_value if clip_value > 0 else 1.0
        np.clip(x, -clip_value, clip_value, out=x)
        x /= scale
        return x.astype(np.float32, copy=False)

    if mode == "trace":
        scale_per_trace = _chunked_trace_percentile(x, clip_percentile)
        scale_safe = np.where(scale_per_trace > 0, scale_per_trace, 1.0)
        clip_lo = -scale_per_trace[:, None]
        clip_hi = scale_per_trace[:, None]
        np.minimum(np.maximum(x, clip_lo, out=x), clip_hi, out=x)
        x /= scale_safe[:, None]
        zero_mask = scale_per_trace <= 0
        if zero_mask.any():
            x[zero_mask, :] = 0.0
        if verbose:
            nz = scale_per_trace[~zero_mask]
            if nz.size > 0:
                print(
                    f"{_hdr}\nClip value (per-trace, |x| {clip_percentile}% percentile): "
                    f"min={nz.min():.4f}, median={np.median(nz):.4f}, max={nz.max():.4f}, "
                    f"zero-traces={int(zero_mask.sum())}/{len(scale_per_trace)}"
                )
            else:
                print(f"{_hdr}\nPer-trace norm: all traces are zero ({len(scale_per_trace)} traces)")
        return x.astype(np.float32, copy=False)

    raise ValueError(f"norm_mode must be 'file' or 'trace', got {mode!r}")

def read_segy_trace_shape(segy_path: str) -> tuple[int, int]:
    """Read only volume header/structure, without loading full trace data."""
    with segyio.open(segy_path, "r", strict=False, ignore_geometry=True) as f:
        return int(f.tracecount), int(len(f.samples))

def write_pred_mask_segy(
    template_sgy_path: str,
    mask_traces_first: np.ndarray,
    out_sgy_path: str,
) -> None:
    """Copy the template SEGY's volume header and trace headers, replacing trace data with mask."""
    arr = np.asarray(mask_traces_first, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"mask should be 2D, current shape={arr.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(out_sgy_path)) or ".", exist_ok=True)
    shutil.copy2(template_sgy_path, out_sgy_path)
    with segyio.open(out_sgy_path, "r+", strict=False, ignore_geometry=True) as f:
        ntr, ns = f.tracecount, len(f.samples)
        if arr.shape[0] != ntr or arr.shape[1] != ns:
            raise ValueError(
                f"predicted mask shape {arr.shape} does not match template SEGY: "
                f"need ({ntr}, {ns}), template={template_sgy_path}"
            )
        f.trace.raw[:] = arr
