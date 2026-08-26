"""Self-contained inference: seismic SEGY -> predicted first-break mask SEGY + picks."""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from scipy.ndimage import zoom as ndimage_zoom
from tqdm import tqdm

from data_utils import (
    _apply_coord_bounds,
    _normalize_seismic,
    read_segy_data,
    read_segy_geom_per_trace,
    read_segy_trace_shape,
    write_pred_mask_segy,
)
from model import (
    create_spatial_first_break_transformer_geom,
)
from picking import (
    extract_picks_from_prediction,
)

SHOT_STRIDE = 64
HR_AT_THRESHOLDS = (1, 3, 5, 7, 9)

def _to_binary_mask_float(label_arr):
    x = np.nan_to_num(np.asarray(label_arr, dtype=np.float32), nan=0.0)
    if x.size == 0:
        return x
    if x.max() > 1.0 + 1e-6 or x.min() < -1e-6:
        x = (x > 0.5).astype(np.float32)
    else:
        x = (x >= 0.5).astype(np.float32)
    return np.clip(x, 0.0, 1.0, out=x)

def _per_trace_label_validity(label_2d: np.ndarray) -> np.ndarray:
    s = label_2d.sum(axis=1, dtype=np.float64)
    return (s > 0.5).astype(np.float32)

def compute_pick_metrics(pred_samples, gt_samples, supervised, hr_thresholds=HR_AT_THRESHOLDS):
    pred_samples = np.asarray(pred_samples, dtype=np.float64).reshape(-1)
    gt_samples = np.asarray(gt_samples, dtype=np.float64).reshape(-1)
    sup = np.asarray(supervised, dtype=np.float64).reshape(-1) > 0.5
    eval_mask = sup & np.isfinite(pred_samples) & np.isfinite(gt_samples)
    n_sup = int(sup.sum())
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        out = {"n_supervised_traces": n_sup, "n_eval_traces": 0}
        for k in hr_thresholds:
            out[f"HR@{k}px"] = float("nan")
        out.update({"RMSE": float("nan"), "MAE": float("nan"), "MBE": float("nan")})
        return out

    pred_e = pred_samples[eval_mask]
    gt_e = gt_samples[eval_mask]
    err = pred_e - gt_e
    abs_err = np.abs(err)
    out = {"n_supervised_traces": n_sup, "n_eval_traces": n_eval,
           "RMSE": float(np.sqrt(np.mean(err ** 2))),
           "MAE": float(np.mean(abs_err)),
           "MBE": float(np.mean(err))}
    for k in hr_thresholds:
        out[f"HR@{k}px"] = float(np.mean(abs_err <= k))
    return out

def write_metrics_txt(path: str, metrics: dict, extra_lines: list[str] | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines = []
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("# First-break pick metrics (sample index units unless noted)")
    for key in ("n_supervised_traces", "n_eval_traces",
                "HR@1px", "HR@3px", "HR@5px", "HR@7px", "HR@9px",
                "RMSE", "MAE", "MBE", "RMSE_sec", "MAE_sec", "MBE_sec"):
        if key in metrics:
            lines.append(f"{key}\t{metrics[key]}")
    for k, v in metrics.items():
        if k in ("n_supervised_traces", "n_eval_traces",
                 "HR@1px", "HR@3px", "HR@5px", "HR@7px", "HR@9px",
                 "RMSE", "MAE", "MBE"):
            continue
        lines.append(f"{k}\t{v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def gt_picks_from_binary_mask_rows(label_bin: np.ndarray) -> np.ndarray:
    """Binary mask (n_tr, nt): per-trace index of first sample >= 0.5; nan if none."""
    lb = np.asarray(label_bin, dtype=np.float64) >= 0.5
    has = lb.any(axis=1)
    out = np.full(lb.shape[0], np.nan, dtype=np.float64)
    if np.any(has):
        out[has] = np.argmax(lb[has], axis=1).astype(np.float64)
    return out

def build_pred_mask_traces_time(merged_probs, picks_samples, *, style="pick", prob_threshold=0.5):
    """Build (n_traces, n_time) predicted mask.

    style:
      pick -- 1 from pick sample onward; non-finite = all zeros
      prob -- per-point binarization: merged > prob_threshold
    """
    merged_probs = np.asarray(merged_probs, dtype=np.float32)
    if merged_probs.ndim != 2:
        raise ValueError(f"merged_probs should be 2D (trace x time), got {merged_probs.shape}")
    n_tr, nt = merged_probs.shape
    if style == "prob":
        return (merged_probs > float(prob_threshold)).astype(np.float32)
    p = np.asarray(picks_samples, dtype=np.float64).reshape(-1)
    if p.size != n_tr:
        raise ValueError(f"pick length {p.size} != trace count {n_tr}")
    fb = np.full(n_tr, nt + 1, dtype=np.int32)
    valid = np.isfinite(p)
    fb[valid] = np.clip(np.rint(p[valid]).astype(np.int64), 0, nt - 1)
    t = np.arange(nt, dtype=np.int32)[None, :]
    return (t >= fb[:, None]).astype(np.float32)

def _infer_input_time_steps_from_state_dict(state: dict) -> int | None:
    w = state.get("encoder.initial_proj.weight")
    if w is None:
        return None
    t = getattr(w, "shape", None)
    if t is None or len(t) < 2:
        return None
    return int(t[1])

def align_seismic_time_for_model(seismic_norm: np.ndarray, nt_model: int) -> np.ndarray:
    """Resample (n_trace, nt_data) -> (n_trace, nt_model) via linear interpolation."""
    x = np.asarray(seismic_norm, dtype=np.float32)
    _n, nt_data = x.shape
    if nt_data == nt_model:
        return x
    factor = float(nt_model) / float(nt_data)
    return ndimage_zoom(x, (1.0, factor), order=1, mode="nearest").astype(np.float32, copy=False)

def remap_merged_probs_to_data_time(merged_model: np.ndarray, nt_data: int, nt_model: int) -> np.ndarray:
    """Reverse-resample model-space probs back to data time grid."""
    merged_model = np.asarray(merged_model, dtype=np.float32)
    _n, nm = merged_model.shape
    if nm != nt_model:
        raise ValueError(f"merged time dim {nm} != nt_model={nt_model}")
    if nt_data == nt_model:
        return merged_model
    factor = float(nt_data) / float(nt_model)
    return ndimage_zoom(merged_model, (1.0, factor), order=1, mode="nearest").astype(np.float32, copy=False)

def _inference_shot_starts(n_traces: int, shot_length: int, stride: int) -> np.ndarray:
    if n_traces < shot_length:
        raise ValueError(f"trace count {n_traces} < shot_length {shot_length}")
    starts = []
    s = 0
    while s + shot_length <= n_traces:
        starts.append(s)
        s += stride
    last_start = n_traces - shot_length
    if not starts:
        starts = [last_start]
    elif starts[-1] != last_start and last_start not in starts:
        starts.append(last_start)
    return np.array(sorted(set(starts)), dtype=np.int64)

def run_inference(*, seismic_norm, coords_norm, shot_length, stride,
                  model, device, batch_size, geom2_norm=None) -> np.ndarray:
    """Sliding-window forward, fuse overlapping regions by visit count.

    Returns (n_traces_orig, time_samples) float32 probability map.
    """
    n_orig, ns = seismic_norm.shape
    if coords_norm.shape[0] != n_orig:
        raise ValueError(f"coords trace count {coords_norm.shape[0]} != seismic {n_orig}")
    if geom2_norm is not None and geom2_norm.shape != (n_orig, 2):
        raise ValueError(f"geom2 shape {geom2_norm.shape} != (n_traces, 2)")

    seismic_work = np.ascontiguousarray(seismic_norm, dtype=np.float32)
    coords_work = np.ascontiguousarray(coords_norm, dtype=np.float32)
    geom_work = np.ascontiguousarray(geom2_norm, dtype=np.float32) if geom2_norm is not None else None

    if n_orig < shot_length:
        pad_tr = shot_length - n_orig
        print(f"Note: trace count {n_orig} < shot_length {shot_length}, padding {pad_tr} traces")
        seismic_work = np.pad(seismic_work, ((0, pad_tr), (0, 0)), mode="constant", constant_values=0.0)
        coords_work = np.pad(coords_work, ((0, pad_tr), (0, 0)), mode="edge")
        if geom_work is not None:
            geom_work = np.pad(geom_work, ((0, pad_tr), (0, 0)), mode="edge")

    n_traces = seismic_work.shape[0]
    starts = _inference_shot_starts(n_traces, shot_length, stride)
    sum_prob = np.zeros((n_traces, ns), dtype=np.float32)
    counts = np.zeros((n_traces, ns), dtype=np.int16)

    n_batches = (len(starts) + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        pbar = tqdm(range(0, len(starts), batch_size), total=n_batches,
                     desc="Inference", unit="batch", dynamic_ncols=True)
        for i0 in pbar:
            batch_starts = starts[i0: i0 + batch_size]
            tensors, coord_tensors, geom_tensors = [], [], []

            for st in batch_starts:
                sl = slice(int(st), int(st) + shot_length)
                tensors.append(torch.from_numpy(np.ascontiguousarray(seismic_work[sl])))
                coord_tensors.append(torch.from_numpy(np.ascontiguousarray(coords_work[sl])))
                if geom_work is not None:
                    geom_tensors.append(torch.from_numpy(np.ascontiguousarray(geom_work[sl])))

            inp = torch.stack(tensors, dim=0).to(device)
            crd = torch.stack(coord_tensors, dim=0).to(device)

            if geom_work is not None:
                g2 = torch.stack(geom_tensors, dim=0).to(device)
                logits = model(inp, crd, geom2=g2)
            else:
                logits = model(inp, crd)

            prob = torch.sigmoid(logits).cpu().numpy()

            for b, st in enumerate(batch_starts):
                sl = slice(int(st), int(st) + shot_length)
                sum_prob[sl] += prob[b]
                counts[sl] += 1

    merged = sum_prob / np.maximum(counts, 1)
    return np.asarray(merged, dtype=np.float32)[:n_orig, :]

def _load_json_config(path: str) -> dict | None:
    """Load a JSON config file; returns None if not found or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Loaded config: {path}")
        return cfg
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="First-break picking inference (SEGY -> picks + pred_mask SEGY)"
    )
    parser.add_argument("--segy", required=True, help="Input seismic SEGY path")
    parser.add_argument("--model", required=True, help="Model checkpoint .pth path")
    parser.add_argument("--output", required=True,
                        help="Output stem: produces {output}_picks.txt and {output}_pred_mask.sgy")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu etc.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (cuda:{id})")

    args = parser.parse_args()

    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    _cfg = _load_json_config(_cfg_path) or {}
    _mcfg = _cfg.get("model", {})
    _gcfg = _cfg.get("geom", {})
    _dcfg = _cfg.get("data", {})
    _pcfg = _cfg.get("picking", {})
    _bcfg = _cfg.get("bounds", {})

    output_stem = os.path.abspath(args.output)
    picks_path = output_stem + "_picks.txt"
    pred_mask_path = output_stem + "_pred_mask.sgy"
    metric_path = output_stem + "_metric.txt"

    if args.device is not None:
        device = torch.device(args.device)
    elif args.gpu is not None:
        if not torch.cuda.is_available():
            parser.error("--gpu specified but no CUDA detected")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seismic_data, seismic_coords, ns, sample_interval_sec = read_segy_data(args.segy)
    n_traces = seismic_data.shape[0]
    if seismic_data.shape[1] != ns:
        print(f"Warning: trace length {seismic_data.shape[1]} != bin Samples {ns}, using actual")
        ns = seismic_data.shape[1]

    print(f"SEGY: {args.segy}")
    print(f"Model: {args.model}")
    print(f"Output stem: {output_stem}")
    print(f"Traces: {n_traces}, Time samples: {ns}, Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    clip_percentile = _dcfg.get("clip_percentile", 99.0)
    norm_mode = _dcfg.get("norm_mode", "trace")
    seismic_norm = _normalize_seismic(seismic_data, clip_percentile=clip_percentile,
                                      mode=norm_mode, verbose=True)

    geom_bounds = None
    if _bcfg:
        lo = np.asarray(_bcfg["coord_lo"], dtype=np.float64)
        hi = np.asarray(_bcfg["coord_hi"], dtype=np.float64)
        if "geom_lo" in _bcfg and "geom_hi" in _bcfg:
            geom_bounds = (np.asarray(_bcfg["geom_lo"], dtype=np.float64),
                           np.asarray(_bcfg["geom_hi"], dtype=np.float64))
        print(f"Coord bounds from config.json: lo={lo.tolist()}, hi={hi.tolist()}")
    else:
        lo = np.nanmin(seismic_coords.astype(np.float64), axis=0)
        hi = np.nanmax(seismic_coords.astype(np.float64), axis=0)
        print(f"Coord bounds (single-file min-max): lo={lo.tolist()}, hi={hi.tolist()}")

    coords_norm = _apply_coord_bounds(seismic_coords, (lo, hi)).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ckpt = torch.load(args.model, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        raise RuntimeError("No valid state dict found in checkpoint")

    geom2_norm = None
    if _gcfg.get("use_geom_mlp", False):
        geom_raw = read_segy_geom_per_trace(args.segy)
        if geom_raw.shape[0] != n_traces:
            raise ValueError(f"geom trace count {geom_raw.shape[0]} != seismic {n_traces}")
        if geom_bounds is not None:
            geom2_norm = _apply_coord_bounds(geom_raw, geom_bounds).astype(np.float32)
        else:
            g_lo = np.nanmin(geom_raw, axis=0)
            g_hi = np.nanmax(geom_raw, axis=0)
            geom2_norm = _apply_coord_bounds(geom_raw, (g_lo, g_hi)).astype(np.float32)

    nt_data = int(ns)
    nt_model = _infer_input_time_steps_from_state_dict(state)
    if nt_model is None:
        nt_model = nt_data
        print("Could not infer time samples from checkpoint, using data time dimension.")
    elif nt_model != nt_data:
        print(f"Training time samples={nt_model}, data={nt_data}: interpolating both ways.")

    pos_encoding = _mcfg.get("pos_encoding", "rope")
    d_model = _mcfg.get("d_model", 1024)
    n_heads = _mcfg.get("n_heads", 8)
    n_stages = _mcfg.get("n_stages", 3)
    d_ff = _mcfg.get("d_ff", 2048)

    use_geom_mlp = _gcfg.get("use_geom_mlp", False)
    geom_mlp_hidden = _gcfg.get("geom_mlp_hidden", 128)
    use_adaln = _gcfg.get("use_adaln", False) and use_geom_mlp
    use_multistage_geom = _gcfg.get("use_multistage_geom", False) and use_geom_mlp
    use_geom_attn_bias = _gcfg.get("use_geom_attn_bias", False) and use_geom_mlp
    geom_attn_bias_temperature = _gcfg.get("geom_attn_bias_temperature", 1.0)
    use_global_trace_context = _gcfg.get("use_global_trace_context", False)

    model = create_spatial_first_break_transformer_geom(
        input_dim=nt_model,
        d_model=d_model, n_heads=n_heads, n_stages=n_stages, d_ff=d_ff,
        coord_dim=_mcfg.get("coord_dim", 4), pos_encoding=pos_encoding,
        use_geom_mlp=use_geom_mlp, geom_mlp_hidden=geom_mlp_hidden,
        use_adaln=use_adaln, use_multistage_geom=use_multistage_geom,
        use_geom_attn_bias=use_geom_attn_bias,
        use_global_trace_context=use_global_trace_context,
        geom_attn_bias_temperature=geom_attn_bias_temperature,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Note: checkpoint missing keys (using default init): {missing}")
    if unexpected:
        print(f"Note: checkpoint has unexpected keys (ignored): {unexpected}")
    model.to(device)

    shot_length = _dcfg.get("shot_length", 256)
    stride = _dcfg.get("stride", SHOT_STRIDE)
    seismic_in = align_seismic_time_for_model(seismic_norm, nt_model)
    merged_model = run_inference(seismic_norm=seismic_in, coords_norm=coords_norm,
                                 shot_length=shot_length, stride=stride,
                                 model=model, device=device, batch_size=args.batch_size,
                                 geom2_norm=geom2_norm)
    merged = remap_merged_probs_to_data_time(merged_model, nt_data, nt_model)

    pick_method = _pcfg.get("method", "npp")
    pick_threshold = _pcfg.get("threshold", 0.5)
    npp_anchor = _pcfg.get("npp_anchor", "earliest")

    pr_map = merged.T
    picks_samples = extract_picks_from_prediction(
        pr_map, threshold=pick_threshold, dt=1.0, npp_anchor=npp_anchor,
    )

    segy_abs = os.path.abspath(args.segy)
    cand_mask = os.path.splitext(segy_abs)[0] + "_mask.sgy"

    if os.path.isfile(cand_mask):
        print(f"Reading label mask: {cand_mask} ...")
        label_data, _, _ns_mask, _ = read_segy_data(cand_mask)
        if label_data.shape == seismic_data.shape:
            label_bin = _to_binary_mask_float(label_data)
            supervised = _per_trace_label_validity(label_bin)

            if n_traces >= 50000:
                gt_samples = gt_picks_from_binary_mask_rows(label_bin)
            else:
                gt_map = label_bin.T
                gt_samples = extract_picks_from_prediction(
                    gt_map, threshold=pick_threshold, dt=1.0, npp_anchor=npp_anchor,
                )

            metrics = compute_pick_metrics(picks_samples, gt_samples, supervised)
            if sample_interval_sec and metrics["n_eval_traces"] > 0:
                s = float(sample_interval_sec)
                metrics["RMSE_sec"] = metrics["RMSE"] * s
                metrics["MAE_sec"] = metrics["MAE"] * s
                metrics["MBE_sec"] = metrics["MBE"] * s

            header = [f"segy\t{args.segy}", f"mask\t{cand_mask}", f"model\t{args.model}",
                      f"pick_method\t{pick_method}", f"npp_anchor\t{npp_anchor}"]
            if sample_interval_sec:
                header.append(f"# sample_interval_sec\t{sample_interval_sec}")
            write_metrics_txt(metric_path, metrics, extra_lines=header)
            print(f"Metrics -> {metric_path} | supervised={metrics['n_supervised_traces']} eval={metrics['n_eval_traces']}")
        else:
            print(f"Warning: mask shape {label_data.shape} != seismic {seismic_data.shape}, skipping metrics")

    template_sgy = cand_mask if os.path.isfile(cand_mask) else args.segy
    tmpl_ntr, tmpl_ns = read_segy_trace_shape(template_sgy)
    if tmpl_ntr != merged.shape[0] or tmpl_ns != merged.shape[1]:
        raise ValueError(f"Template SEGY shape ({tmpl_ntr}, {tmpl_ns}) != predicted {merged.shape}")
    pred_mask_arr = build_pred_mask_traces_time(merged, picks_samples)
    print(f"Writing predicted mask SEGY -> {pred_mask_path} ...")
    write_pred_mask_segy(template_sgy, pred_mask_arr, pred_mask_path)
    print(f"Predicted mask SEGY written (template={template_sgy})")

    picks = np.asarray(picks_samples, dtype=np.float64).reshape(-1)
    os.makedirs(os.path.dirname(os.path.abspath(picks_path)) or ".", exist_ok=True)
    np.savetxt(picks_path, picks, fmt="%.8e")
    print(f"Picks -> {picks_path}  ({n_traces} traces)")

if __name__ == "__main__":
    main()
