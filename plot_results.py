"""
SEG-Y shot/CDP gather visualization with first-break picks.

Reads a SEG-Y file and a picks txt file, groups traces by (shot id, CDP),
and generates seismic + pick-overlay images for each CDP group.

Examples:
    python plot_results.py -d test_results --sgy test_data/fig4_data.sgy --picks test_results/fig4_pred_picks.txt
    python plot_results.py -d test_results --sgy test_data/fig4_data.sgy --picks test_results/fig4_pred_picks.txt --line 101
    python plot_results.py -d /path/to/data --shot 20081540
"""

import argparse
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BYTE_POS = {"shot_no": 9, "cdp": 21, "rec_x": 81, "rec_y": 85}

def seismic(iop=1):
    """
    Seismic colormap for Python

    Parameters:
        iop: int
            1 = min brown, zero white, max black
            2 = min red, zero white, max black
            3 = min blue, zero white, max red
            4 = custom (less used)

    Returns:
        M: ListedColormap
            Matplotlib colormap object
    """
    N = 40
    L = 40
    size_total = 128

    if iop == 1:
        u1 = np.concatenate(
            [
                0.5 * np.ones(N),
                np.linspace(0.5, 1, size_total - N),
                np.linspace(1, 0, size_total - N),
                np.zeros(N),
            ]
        )
        u2 = np.concatenate(
            [
                0.25 * np.ones(N),
                np.linspace(0.25, 1, size_total - N),
                np.linspace(1, 0, size_total - N),
                np.zeros(N),
            ]
        )
        u3 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0, 1, size_total - N),
                np.linspace(1, 0, size_total - N),
                np.zeros(N),
            ]
        )
    elif iop == 2:
        u1 = np.concatenate(
            [
                np.ones(N),
                np.linspace(1, 1, size_total - N),
                np.linspace(1, 0, size_total - N),
                np.zeros(N),
            ]
        )
        u2 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0, 1, size_total - N),
                np.linspace(1, 0, size_total - N),
                np.zeros(N),
            ]
        )
        u3 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0, 1, size_total - N),
                np.linspace(1, 0, size_total - N),
                np.zeros(N),
            ]
        )
    elif iop == 3:
        u1 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0, 1, size_total - N - L // 2),
                np.ones(L),
                np.linspace(1, 0.5, size_total - L // 2),
            ]
        )
        u2 = np.concatenate(
            [
                np.zeros(N),
                np.linspace(0, 1, size_total - N - L // 2),
                np.ones(L),
                np.linspace(1, 0, size_total - N - L // 2),
                np.zeros(N),
            ]
        )
        u3 = np.concatenate(
            [
                np.linspace(0.5, 1, size_total - L // 2),
                np.ones(L),
                np.linspace(1, 0, size_total - N - L // 2),
                np.zeros(N),
            ]
        )
    elif iop == 4:
        u1 = np.concatenate([np.linspace(1, 1, 128), np.linspace(1, 0, 128)])
        u2 = np.concatenate([np.linspace(0, 1, 128), np.linspace(1, 0, 128)])
        u3 = np.concatenate([np.linspace(0, 1, 128), np.linspace(1, 1, 128)])
    else:
        raise ValueError("iop must be 1,2,3 or 4")

    M = np.vstack([u1, u2, u3]).T
    return ListedColormap(M)

def resolve_cmap(cmap_spec):
    """Resolve --cmap. Supports Matplotlib names and custom seismic(1..4)."""
    cmap_spec = str(cmap_spec).strip()

    if cmap_spec.lower().startswith("seismic(") and cmap_spec.endswith(")"):
        raw_iop = cmap_spec[cmap_spec.find("(") + 1 : -1].strip()
        try:
            iop = int(raw_iop)
        except ValueError as exc:
            raise ValueError(f"Invalid custom seismic cmap spec: {cmap_spec}") from exc
        print(f"Using custom seismic({iop}) colormap")
        return seismic(iop)

    if cmap_spec in plt.colormaps():
        return cmap_spec

    if "(" in cmap_spec and cmap_spec.endswith(")"):
        base_name = cmap_spec.split("(", 1)[0].strip()
        if base_name in plt.colormaps():
            print(f"Using cmap '{base_name}' from input '{cmap_spec}'")
            return base_name

    examples = "Greys, gray, seismic, seismic_r, RdBu, bwr, viridis, seismic(1), seismic(2), seismic(3)"
    raise ValueError(f"Invalid cmap '{cmap_spec}'. Use a Matplotlib colormap name, e.g. {examples}.")

def i32be(buf, pos1b):
    return struct.unpack(">i", buf[pos1b - 1 : pos1b + 3])[0]

def read_segy_and_picks(data_dir, sgy_path=None, picks_path=None):
    """Read SEG-Y and picks txt, grouped by (shot id, CDP)."""
    data_dir = Path(data_dir)

    if sgy_path:
        sgy_path = Path(sgy_path)
    else:
        sgy_path = data_dir / "seismic.sgy"
    if not sgy_path.exists():
        raise FileNotFoundError(f"SEG-Y not found: {sgy_path}")

    with open(sgy_path, "rb") as f:
        f.seek(3200)
        bh = f.read(400)
        ns = struct.unpack(">H", bh[20:22])[0]
        fmt = struct.unpack(">H", bh[24:26])[0]
        dt_us = struct.unpack(">H", bh[16:18])[0]

    bps = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(fmt, 4)
    dtype_str = {5: ">f4", 3: ">i2", 2: ">i4", 8: ">i1"}.get(fmt, ">f4")

    shot_traces = {}
    global_tidx = 0
    with open(sgy_path, "rb") as f:
        f.seek(3600)
        while True:
            h = f.read(240)
            if len(h) < 240:
                break
            sn = i32be(h, BYTE_POS["shot_no"])
            cdp = i32be(h, BYTE_POS["cdp"])
            tn = struct.unpack(">H", h[114:116])[0] or ns
            tr = np.frombuffer(f.read(tn * bps), dtype=dtype_str).astype(np.float32)
            shot_traces.setdefault(sn, []).append((global_tidx, cdp, tr))
            global_tidx += 1

    print(
        f"Read {len(shot_traces)} shots, {global_tidx} traces, "
        f"ns={ns}, dt={dt_us} us ({dt_us / 1000:.1f} ms)"
    )

    picks_data = {"pred": {}}
    if picks_path:
        picks_candidates = [Path(picks_path)]
    else:
        picks_candidates = sorted(data_dir.glob("picks*.txt"))
    if picks_candidates:
        picks_path = picks_candidates[0]
        if len(picks_candidates) > 1:
            print(
                f"  Multiple picks txt files found: {[p.name for p in picks_candidates]}; "
                f"using {picks_path.name}"
            )
        with open(picks_path, "r") as f:
            first_line = ""
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    first_line = line
                    break
        parts = first_line.split()
        if len(parts) == 1:
            picks = {}
            with open(picks_path, "r") as f:
                for tidx, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    val = float(line.split()[0])
                    if val >= 0:
                        picks[tidx] = val
            picks_data["pred"] = picks
            print(f"  Loaded {picks_path.name} pred (single-column): {len(picks)} valid picks")
        else:
            picks_data = {"label": {}, "geoformer": {}, "atten_unet": {}, "hunet": {}}
            raw = {"label": {}, "geoformer": {}, "atten_unet": {}, "hunet": {}}
            pick_columns = [
                ("label", 1),
                ("geoformer", 2),
                ("atten_unet", 3),
                ("hunet", 4),
            ]
            with open(picks_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    tidx = int(parts[0])
                    for key, col in pick_columns:
                        if len(parts) <= col:
                            continue
                        val = float(parts[col])
                        if val >= 0:
                            raw[key][tidx] = val
            for key in raw:
                print(f"  Loaded {picks_path.name} {key}: {len(raw[key])} valid picks")
                picks_data[key] = raw[key]
    else:
        print("  WARNING: no picks*.txt found")

    gathers = {}
    for sn, traces in shot_traces.items():
        cdp_groups = {}
        for i, (_tidx, cdp, _tr) in enumerate(traces):
            cdp_groups.setdefault(cdp, []).append(i)

        sn_gathers = []
        for cdp, idx_list in cdp_groups.items():
            idx_list.sort(key=lambda i: traces[i][0])
            traces_2d = np.array([traces[i][2] for i in idx_list], dtype=np.float32)
            tidx_sorted = np.array([traces[i][0] for i in idx_list], dtype=np.int64)
            trace_order = np.arange(len(idx_list), dtype=np.float64)
            sn_gathers.append((cdp, trace_order, tidx_sorted, traces_2d))

        sn_gathers.sort(key=lambda x: x[0])
        gathers[sn] = sn_gathers

    total_lines = sum(len(g) for g in gathers.values())
    print(f"Total CDP groups: {total_lines}")
    return gathers, ns, dt_us, picks_data

def plot_seismic_with_picks(
    ax,
    dist,
    t,
    traces_2d,
    tidx_sorted,
    picks_dict,
    vmin,
    vmax,
    dt_us,
    cmap,
):
    """Plot seismic traces with optional red first-break pick markers."""
    im = ax.imshow(
        traces_2d.T,
        aspect="auto",
        origin="upper",
        extent=[dist[0], dist[-1], t[-1], t[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    pick_x, pick_t = [], []
    for i, tidx in enumerate(tidx_sorted):
        tidx_key = int(tidx)
        if tidx_key in picks_dict:
            pick_x.append(dist[i])
            pick_t.append(picks_dict[tidx_key] * dt_us / 1e6)

    if pick_x:
        ax.scatter(pick_x, pick_t, c="red", s=10, marker="o", edgecolors="none", zorder=5)

    return im

def plot_line_gather_all(
    trace_order,
    tidx_sorted,
    traces_2d,
    ns,
    dt_us,
    picks_data,
    out_dir,
    gain=1.0,
    ds=1,
    dpi=150,
    fig_width=8.0,
    fig_height=12.0,
    vmin=None,
    cmap="Greys",
    tight=True,
    pad_inches=0.05,
):
    """Generate all output images for one CDP gather directly in out_dir."""
    traces_2d = traces_2d * gain
    n_tr = traces_2d.shape[0]

    if ds > 1:
        nz = ns // ds
        traces_2d = traces_2d[:, : nz * ds].reshape(n_tr, nz, ds).mean(axis=2)
    else:
        nz = ns

    t = np.arange(nz) * ds * dt_us / 1e6
    dist = trace_order

    if vmin is None:
        vmax = np.std(traces_2d) * 2.5
        vmin = -vmax
    else:
        vmin = float(vmin)
        vmax = -vmin

    if "pred" in picks_data:
        variants = [
            ("seismic", "Seismic", {}),
            ("pred", "Prediction", picks_data.get("pred", {})),
        ]
    else:
        variants = [
            ("seismic", "Seismic", {}),
            ("label", "Label (GT)", picks_data.get("label", {})),
            ("atten_unet", "Atten_UNet", picks_data.get("atten_unet", {})),
            ("geoformer", "Geoformer", picks_data.get("geoformer", {})),
            ("hunet", "HUNet", picks_data.get("hunet", {})),
        ]

    for fname_stem, subtitle, pick_dict in variants:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plot_seismic_with_picks(
            ax,
            dist,
            t,
            traces_2d,
            tidx_sorted,
            pick_dict,
            vmin,
            vmax,
            dt_us,
            cmap,
        )
        ax.set_xlabel("Trace number", fontsize=24)
        ax.set_ylabel("Time (s)", fontsize=24)
        ax.set_title(subtitle, fontsize=24)
        ax.tick_params(labelsize=23)
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs.update({"bbox_inches": "tight", "pad_inches": pad_inches})
        fig.savefig(out_dir / f"{fname_stem}.png", **save_kwargs)
        plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Plot SEG-Y shot/CDP gathers with first-break picks")
    p.add_argument("-d", "--data-dir", required=True, help="data directory containing seismic.sgy and picks txt")
    p.add_argument("-o", "--outdir", default=None, help="output directory for images (default: <data-dir>)")
    p.add_argument("-g", "--gain", type=float, default=1.0, help="amplitude gain factor (default: 1.0)")
    p.add_argument("--ds", type=int, default=1, help="time downsampling factor (default: 1)")
    p.add_argument("--shot", type=int, default=None, help="only process the specified shot id")
    p.add_argument("--line", type=int, default=None, help="only process the specified CDP id")
    p.add_argument("--cdp-gather", type=int, default=None, help="only redraw the specified CDP_Gather# id")
    p.add_argument("--dpi", type=int, default=150, help="output image DPI (default: 150)")
    p.add_argument("--fig-width", type=float, default=8.0, help="figure width in inches (default: 8)")
    p.add_argument("--fig-height", type=float, default=12.0, help="figure height in inches (default: 12)")
    p.add_argument("--vmin", type=float, default=None, help="image vmin; vmax is set to -vmin")
    p.add_argument(
        "--cmap",
        default="Greys",
        help="Matplotlib colormap name or custom seismic(1..4) (default: Greys)",
    )
    p.add_argument("--no-tight", action="store_true", help="disable tight bbox image saving")
    p.add_argument("--pad-inches", type=float, default=0.05, help="padding for tight bbox in inches (default: 0.05)")
    p.add_argument("--sgy", default=None, help="path to SEG-Y file (default: <data_dir>/seismic.sgy)")
    p.add_argument("--picks", default=None, help="path to picks txt file (default: <data_dir>/picks*.txt)")
    a = p.parse_args()
    a.cmap = resolve_cmap(a.cmap)

    if a.line is not None and a.cdp_gather is not None and a.line != a.cdp_gather:
        print(f"--line ({a.line}) and --cdp-gather ({a.cdp_gather}) do not match")
        sys.exit(1)
    target_cdp = a.cdp_gather if a.cdp_gather is not None else a.line
    if target_cdp is None:
        target_cdp = 5540

    data_dir = Path(a.data_dir)
    if a.outdir:
        out_dir = Path(a.outdir)
    else:
        out_dir = data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gathers, ns, dt_us, picks_data = read_segy_and_picks(data_dir, sgy_path=a.sgy, picks_path=a.picks)

    shot_list = sorted(gathers.keys())
    if a.shot is not None:
        if a.shot not in gathers:
            print(f"Shot {a.shot} not found. Available shots: {shot_list}")
            sys.exit(1)
        shot_list = [a.shot]

    total = 0
    for sn in shot_list:
        for cdp, trace_order, tidx_sorted, traces_2d in gathers[sn]:
            if target_cdp is not None and cdp != target_cdp:
                continue
            total += 1
            n_tr = traces_2d.shape[0]

            info = f"CDP#{cdp} Shot#{sn} {n_tr}traces dt={dt_us / 1000:.1f}ms"
            print(f"[{total}] {info} ...", end=" ", flush=True)
            plot_line_gather_all(
                trace_order,
                tidx_sorted,
                traces_2d,
                ns,
                dt_us,
                picks_data,
                out_dir,
                gain=a.gain,
                ds=a.ds,
                dpi=a.dpi,
                fig_width=a.fig_width,
                fig_height=a.fig_height,
                vmin=a.vmin,
                cmap=a.cmap,
                tight=not a.no_tight,
                pad_inches=a.pad_inches,
            )
            print("done")

    if total == 0:
        print(f"No matching CDP gather found for target {target_cdp}")
    print(f"\nDone: {total} CDP gather(s), output dir: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
