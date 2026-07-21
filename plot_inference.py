#!/usr/bin/env python
"""plot_inference.py -- make ADAF validation maps from PRE-COMPUTED inference output.

This reads the NetCDF files written by aschein's ``inference_parallel.py`` -- which
already append the un-normalized model fields to each input file -- so it needs NO
torch, NO checkpoint, NO stats.csv, and NO config that matches the checkpoint. That
whole apparatus (and the 2288x1344-vs-2294x1356 / t,q-vs-q,t / 57-missing-keys
matching headaches) lived only to *produce* these fields; here we just plot them.

Expected variables in each output .nc (see inference_parallel.py):
    output_<var>            reconstructed analysis (residual + HRRR bg, un-normalized)
    output_residual_<var>   raw model residual (the model's innovation over HRRR)
    rtma_<var>              truth analysis            hrrr_<var>  HRRR background
    lat / lon (coords)      obs_source (0/1/2, 2=METAR)   obs_mask
where <var> in {t, q, u10, v10}.

Two modes
---------
1. Single file  (--input FILE):
     output      raw model analysis            output_<var>
     innovation  model residual over HRRR      output_residual_<var>   (centered cbar)
     error       model error vs truth          output_<var> - rtma_<var>
                 background error vs truth      hrrr_<var> - rtma_<var>  (centered cbar)

   Pass --error-limit to put the innovation and BOTH error panels of a variable on ONE
   symmetric scale -- required if the maps are meant to show model error < HRRR error,
   since per-panel autoscale hides exactly that difference (see --error-limit).

2. Compare      (--input FILE --compare FILE2, or --compare-dir A B):
     difference  output_<var>(A) - output_<var>(B)   (centered cbar)
   Use this to diff two inference runs -- e.g. all-obs vs --exclude_metar, or
   the Blosc-faithfulness check: inference on data_blosc_combined vs original zlib,
   where the difference map should be ~0 everywhere.

Examples
--------
    # all single-file maps for one cycle, into ./Plots
    python plot_inference.py \
        --input /scratch3/BMC/wrfruc/aschein/ADAF_RTMA/test_output/2023-01-01_00.nc

    # Blosc-vs-zlib faithfulness difference (should be ~0)
    python plot_inference.py --compare-dir OUT_BLOSC OUT_ZLIB \
        --types difference --tag blosc_vs_zlib
"""
import argparse
import os
import re

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")          # headless compute/login nodes
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib.pyplot as plt
import xarray as xr

from utils.misc_functions import model_label

try:                                                # so Blosc-compressed input reads
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

VARS = ["t", "q", "u10", "v10"]
UNITS = {"t": "K", "q": "kg/kg", "u10": "m/s", "v10": "m/s"}


def _extent(ds):
    lon = np.asarray(ds.coords["lon"].values)
    lat = np.asarray(ds.coords["lat"].values)
    return [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]


def cycle_label(path):
    """'2023-01-15_00.nc' -> 'Valid 2023-01-15 00:00 UTC'. None if the name isn't a cycle."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})$", stem)
    if not m:
        return None
    y, mo, d, h = m.groups()
    return f"Valid {y}-{mo}-{d} {h}:00 UTC"


def plot_field(arr, extent, *, title, cbar_label, savepath,
               style="normal", vlim=None, cmap="bwr", subtitle=None):
    """imshow one 2D field with georeferenced extent (matches aschein's convention)."""
    arr = np.asarray(arr)
    if style == "centered":                         # symmetric about 0, robust to outliers
        m = vlim if vlim is not None else float(np.nanpercentile(np.abs(arr), 99.5))
        vmin, vmax = -m, m
    elif style == "extreme":                        # user-forced symmetric limit
        vmin, vmax = -vlim, vlim
    else:                                           # normal: matplotlib autoscale
        vmin = vmax = None

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(arr, origin="lower", cmap=cmap, extent=extent,
                   aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} | min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}",
                 pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.5, color="0.35")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.016)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {savepath}")


def west_seam(ds, override=None):
    """Columns [0, seam) are the RTMA _wexp coarse fill, not an analysis.

    The model is trained with them zeroed (mask_outside_grid184), so nothing there is
    meaningful -- but it is ~3.7% of the grid with ~3x the interior error RMS on the
    wind vars, which both drew a bogus far-west blob and inflated every `centered`
    colorbar (u10 error: 2.90 -> 4.44). Blank it. Inference now writes NaN into the
    strip itself; this keeps older output files and the hrrr-vs-RTMA baseline panel
    (both real fields, but coarse fill there) on the same domain.

    Auto = 149 on the 1356x2294 grid the seam is calibrated for, else 0. `--mask-west`
    overrides; `--mask-west 0` restores the old full-width plots.
    """
    if override is not None:
        return int(override)
    return 149 if (ds.sizes.get("y"), ds.sizes.get("x")) == (1356, 2294) else 0


def _get(ds, name, seam=0):
    if name not in ds:
        raise KeyError(f"'{name}' not in {list(ds.data_vars)[:8]}... "
                       "-- is this a pre-computed inference output file?")
    arr = np.asarray(ds[name].values).astype(np.float32)
    if seam:
        arr[..., :seam] = np.nan
    return arr


def _subtitle(*parts):
    return "  ·  ".join(p for p in parts if p) or None


def single_file(path, out_dir, types, variables, tag, error_limit, label=None,
                mask_west=None):
    ds = xr.open_dataset(path, engine="netcdf4")
    # Blank, don't slice: every column is kept, so the lon/lat extent below stays the
    # array's own bounding box and no re-registration is needed.
    seam = west_seam(ds, mask_west)
    extent = _extent(ds)
    stamp = tag or os.path.splitext(os.path.basename(path))[0]
    valid = _subtitle(cycle_label(path), model_label(ds, label))
    for v in variables:
        u = UNITS[v]
        # One symmetric limit per variable, shared by the innovation AND both error
        # panels. Per-panel autoscale gives each map its OWN colorbar, so "model error
        # vs HRRR error" cannot be read off the colors at all -- the bigger HRRR error
        # just gets a wider scale and ends up looking the same. `output` keeps its own
        # autoscale: it is an absolute field, not a difference, so a shared
        # difference-limit is meaningless for it.
        lim = error_limit.get(v)
        style = "extreme" if lim else "centered"
        if "output" in types:
            plot_field(_get(ds, f"output_{v}", seam), extent,
                       title=f"output_{v} (analysis)",
                       cbar_label=f"{v} [{u}]",
                       savepath=os.path.join(out_dir, f"output_output_{v}_{stamp}.png"),
                       style="normal", subtitle=valid)
        if "innovation" in types:
            plot_field(_get(ds, f"output_residual_{v}", seam), extent,
                       title=f"innovation output_{v} (residual over HRRR)",
                       cbar_label=f"{v} residual [{u}]",
                       savepath=os.path.join(out_dir, f"innovation_output_{v}_{stamp}.png"),
                       style=style, vlim=lim, subtitle=valid)
        if "error" in types:
            plot_field(_get(ds, f"output_{v}", seam) - _get(ds, f"rtma_{v}", seam), extent,
                       title=f"error output_{v} (model - RTMA)",
                       cbar_label=f"{v} error [{u}]",
                       savepath=os.path.join(out_dir, f"error_output_{v}_{stamp}.png"),
                       style="extreme" if lim else "centered", vlim=lim, subtitle=valid)
            plot_field(_get(ds, f"hrrr_{v}", seam) - _get(ds, f"rtma_{v}", seam), extent,
                       title=f"error hrrr_{v} (HRRR - RTMA)",
                       cbar_label=f"{v} error [{u}]",
                       savepath=os.path.join(out_dir, f"error_hrrr_{v}_{stamp}.png"),
                       style="extreme" if lim else "centered", vlim=lim, subtitle=valid)
    ds.close()


def compare(path_a, path_b, out_dir, variables, tag, label=None, mask_west=None):
    """difference of output_<var> between two runs (A - B). Expect ~0 for Blosc vs zlib."""
    a = xr.open_dataset(path_a, engine="netcdf4")
    b = xr.open_dataset(path_b, engine="netcdf4")
    seam = west_seam(a, mask_west)
    extent = _extent(a)
    stamp = tag or "compare"
    la, lb = model_label(a, label), model_label(b)
    pair = f"A={la} - B={lb}" if (la and lb) else (la or lb)
    valid = _subtitle(cycle_label(path_a), pair)
    for v in variables:
        diff = _get(a, f"output_{v}", seam) - _get(b, f"output_{v}", seam)
        amax = float(np.nanmax(np.abs(diff)))
        print(f"  output_{v}: max|A-B| = {amax:.3e} {UNITS[v]}")
        plot_field(diff, extent,
                   title=f"difference output_{v} (A - B)",
                   cbar_label=f"{v} diff [{UNITS[v]}]",
                   savepath=os.path.join(out_dir, f"difference_output_{v}_{stamp}.png"),
                   style="centered", subtitle=valid)
    a.close()
    b.close()


def _match(directory, basename):
    p = os.path.join(directory, basename)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{basename} not found in {directory}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="a pre-computed inference output .nc (single-file modes)")
    ap.add_argument("--compare", help="second output .nc; plot output_<var> difference vs --input")
    ap.add_argument("--compare-dir", nargs=2, metavar=("DIR_A", "DIR_B"),
                    help="diff every file present in both dirs (matched by basename)")
    ap.add_argument("--output-dir", default="Plots")
    ap.add_argument("--types", nargs="+", default=["output", "innovation", "error"],
                    choices=["output", "innovation", "error", "difference"])
    ap.add_argument("--vars", nargs="+", default=VARS, choices=VARS, dest="variables")
    ap.add_argument("--tag", default=None, help="filename suffix (default: input stem)")
    ap.add_argument("--label", default=None,
                    help="model label for the subtitle, e.g. e615 "
                         "(default: read the checkpoint epoch from the file)")
    ap.add_argument("--error-limit", nargs="+", default=[], metavar="VAR=VAL",
                    help="force one symmetric cbar per variable, shared by the innovation "
                         "and BOTH error panels (model-RTMA and HRRR-RTMA) so they are "
                         "directly comparable. e.g. t=4 u10=3.6. `output` is unaffected.")
    ap.add_argument("--mask-west", type=int, default=None, metavar="NCOL",
                    help="blank the westernmost NCOL columns (the RTMA _wexp coarse-fill "
                         "strip the model is trained to ignore). Default: 149 on the "
                         "1356x2294 grid, 0 otherwise. Pass 0 for full-width plots.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    error_limit = {}
    for kv in args.error_limit:
        k, val = kv.split("=")
        error_limit[k] = float(val)

    if args.compare_dir:
        da, db = args.compare_dir
        common = sorted(set(os.listdir(da)) & set(os.listdir(db)))
        common = [f for f in common if f.endswith(".nc")]
        if not common:
            raise SystemExit(f"no common .nc files between {da} and {db}")
        print(f"comparing {len(common)} file(s)")
        for f in common:
            print(f"[{f}]")
            compare(_match(da, f), _match(db, f), args.output_dir, args.variables,
                    tag=args.tag or os.path.splitext(f)[0], label=args.label,
                    mask_west=args.mask_west)
        return

    if not args.input:
        ap.error("--input is required unless --compare-dir is used")

    if args.compare:
        print(f"[compare] {args.input}  -  {args.compare}")
        compare(args.input, args.compare, args.output_dir, args.variables, args.tag,
                label=args.label, mask_west=args.mask_west)
    else:
        print(f"[single] {args.input}")
        single_file(args.input, args.output_dir, args.types, args.variables,
                    args.tag, error_limit, label=args.label, mask_west=args.mask_west)


if __name__ == "__main__":
    main()
