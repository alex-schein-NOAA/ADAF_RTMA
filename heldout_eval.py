#!/usr/bin/env python
"""heldout_eval.py -- held-out-obs skill of the ADAF corrector. NO torch.

Reads the NetCDF written by ``inference_parallel.py`` (the ONE model run), which
already stores every field un-normalized and named by variable, plus:
    heldout_mask   (y,x) int8  1 = this obs was withheld from the model input
    obs_mask       (y,x)       raw complete-station field (0 everywhere on a
                               degenerate year-boundary cycle -> skip)
    sta_<v>        (t,y,x)     station obs; truth = last time slice, sta_<v>[-1]
    output_<v>     (y,x)       model analysis (residual + HRRR background)
    hrrr_<v>       (y,x)       HRRR background (no-model baseline)
    rtma_<v>       (y,x)       RTMA analysis (upper-bound reference)
for <v> in {t, q, u10, v10}.

At every held-out cell that also has a valid obs for variable v, it scores three
sources against the withheld ob -- Model / HRRR / RTMA -- with Pearson corr and
base-unit RMSE, pooled and per hourly cycle. Because inference already writes
base units, no normalization/stats/checkpoint/config is needed here; corr is
scale-invariant and RMSE is already physical, so these numbers are identical to
the old normalized-accumulator eval.

Two stages so re-plotting never touches the GPU/inference again:
    # score inference outputs -> cache + PNGs
    python heldout_eval.py --indir OUT --glob '2023-01-*.nc' \
        --outdir Plots/jan2023 --cache Plots/jan2023/records.npz
    # restyle from the cache alone (seconds, numpy+matplotlib only)
    python heldout_eval.py --from-cache Plots/jan2023/records.npz --outdir Plots/jan2023
"""
import argparse
import glob
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")           # headless login/compute nodes
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# --------------------------------------------------------------------------- #
# Constants. VARS order is the plot/column order only -- fields are read by
# NAME from the .nc, so the old "t,q vs q,t channel-order" hazard is gone.
# q is converted to g/kg (*1000) so every number/axis reads physically.
# --------------------------------------------------------------------------- #
VARS = ["t", "q", "u10", "v10"]
NVAR = len(VARS)
UNIT = {"t": "°C", "q": "g/kg", "u10": "m/s", "v10": "m/s"}
Q_TO_BASE = {"t": 1.0, "q": 1000.0, "u10": 1.0, "v10": 1.0}   # kg/kg -> g/kg for q

# dataviz palette (matches the retired heldout_eval/plots.py so figures are
# visually unchanged): Model / HRRR / RTMA + ink/chrome tokens.
C = {"Model": "#2a78d6", "HRRR": "#1baf7a", "RTMA": "#e0a02e"}
INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


# --------------------------------------------------------------------------- #
# Stage 1: read inference outputs -> per-held-out-point records (base units).
# --------------------------------------------------------------------------- #
def read_records(indir, pattern):
    """Score every held-out point across all matching output files.

    Returns a dict of parallel 1-D arrays (one row per held-out point per var):
        cycle:int32  var:int8(0..3)  truth model hrrr rtma:f32  lat lon:f32
        obs_source:int8
    Degenerate cycles (raw obs_mask all-zero) are skipped, matching the old eval.
    """
    import xarray as xr
    try:
        import hdf5plugin  # noqa: F401  (Blosc-compressed reads)
    except ImportError:
        pass

    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        raise SystemExit(f"no files matched {os.path.join(indir, pattern)}")

    cols = {k: [] for k in
            ("cycle", "var", "truth", "model", "hrrr", "rtma", "lat", "lon", "obs_source")}
    n_used = n_skipped = 0
    for cyc, path in enumerate(files):
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if "heldout_mask" not in ds:
                raise SystemExit(
                    f"{os.path.basename(path)} has no 'heldout_mask' -- run "
                    "inference_parallel.py with --hold_out_obs true first.")
            # Degenerate year-boundary cycle: raw obs_mask all-zero -> the truth
            # slice is a sparse straggler that would poison the pooled stats.
            if "obs_mask" in ds and int(np.asarray(ds["obs_mask"]).sum()) == 0:
                n_skipped += 1
                continue
            held = np.asarray(ds["heldout_mask"].values) == 1
            if not held.any():
                n_skipped += 1
                continue

            lat = np.asarray(ds["lat"].values)
            lon = np.asarray(ds["lon"].values)
            src = (np.asarray(ds["obs_source"].values).astype(np.int8)
                   if "obs_source" in ds else np.zeros(held.shape, np.int8))

            for vi, v in enumerate(VARS):
                truth = np.asarray(ds[f"sta_{v}"].values)[-1]     # last time slice
                sel = held & (truth != 0)                         # valid obs only
                if not sel.any():
                    continue
                k = Q_TO_BASE[v]
                cols["cycle"].append(np.full(int(sel.sum()), cyc, np.int32))
                cols["var"].append(np.full(int(sel.sum()), vi, np.int8))
                cols["truth"].append(truth[sel].astype(np.float32) * k)
                cols["model"].append(np.asarray(ds[f"output_{v}"].values)[sel].astype(np.float32) * k)
                cols["hrrr"].append(np.asarray(ds[f"hrrr_{v}"].values)[sel].astype(np.float32) * k)
                cols["rtma"].append(np.asarray(ds[f"rtma_{v}"].values)[sel].astype(np.float32) * k)
                cols["lat"].append(lat[sel].astype(np.float32))
                cols["lon"].append(lon[sel].astype(np.float32))
                cols["obs_source"].append(src[sel])
        n_used += 1
        if n_used % 50 == 0:
            print(f"  ...{n_used} cycles read", flush=True)

    rec = {k: (np.concatenate(v) if v else np.array([], dtype=np.float32))
           for k, v in cols.items()}
    rec["n_cycles"] = np.int64(n_used)
    print(f"read {n_used} cycles ({n_skipped} skipped), "
          f"{rec['truth'].size} held-out points", flush=True)
    return rec


# --------------------------------------------------------------------------- #
# Metrics (base units already; corr scale-invariant, RMSE physical).
# --------------------------------------------------------------------------- #
def _corr_rmse(pred, truth):
    if pred.size < 2 or pred.std() == 0 or truth.std() == 0:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(pred, truth)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    return corr, rmse


def pooled(rec):
    """{var: {source: (corr, rmse, n)}} pooled over all held-out points."""
    out = {}
    for vi, v in enumerate(VARS):
        m = rec["var"] == vi
        t = rec["truth"][m]
        out[v] = {}
        for s in ("Model", "HRRR", "RTMA"):
            p = rec[{"Model": "model", "HRRR": "hrrr", "RTMA": "rtma"}[s]][m]
            c, r = _corr_rmse(p, t)
            out[v][s] = (c, r, int(t.size))
    return out


def per_cycle(rec, v, metric):
    """{source: 1-D array, one value per cycle} of `metric` in {corr,rmse}."""
    m = rec["var"] == v if isinstance(v, int) else rec["var"] == VARS.index(v)
    cyc = rec["cycle"][m]
    t = rec["truth"][m]
    preds = {s: rec[k][m] for s, k in
             (("Model", "model"), ("HRRR", "hrrr"), ("RTMA", "rtma"))}
    order = np.argsort(cyc, kind="stable")
    cyc_s = cyc[order]
    bounds = np.flatnonzero(np.diff(cyc_s)) + 1
    slices = np.split(np.arange(cyc_s.size), bounds)
    out = {}
    for s in ("Model", "HRRR", "RTMA"):
        p_s, t_s = preds[s][order], t[order]
        vals = []
        for sl in slices:
            pp, tt = p_s[sl], t_s[sl]
            if metric == "corr":
                if pp.size < 3 or pp.std() == 0 or tt.std() == 0:
                    continue
                vals.append(np.corrcoef(pp, tt)[0, 1])
            else:  # rmse
                vals.append(np.sqrt(np.mean((pp - tt) ** 2)))
        out[s] = np.asarray(vals, dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Plots (torch-free; same look as the retired package).
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)


def _save(fig, outpath):
    os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    _plt().close(fig)
    return outpath


def _annot(ax, bars):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=7.5, color=INK,
                    xytext=(0, 2), textcoords="offset points")


def bar_corr(pl, ncyc, outpath):
    plt = _plt()
    srcs = ["Model", "HRRR", "RTMA"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    x = np.arange(NVAR)
    w = 0.8 / len(srcs)
    off = -(len(srcs) - 1) / 2.0
    for i, s in enumerate(srcs):
        vals = [pl[v][s][0] for v in VARS]
        _annot(ax, ax.bar(x + (off + i) * w, vals, w, label=s, color=C[s], zorder=3))
    ax.set_xticks(x); ax.set_xticklabels(VARS, color=INK, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pearson correlation vs held-out obs", color=INK, fontsize=10)
    ax.set_title(f"Analysis skill at held-out stations  (10% hold-out, {ncyc} cycles)",
                 color=INK, fontsize=11, pad=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return _save(fig, outpath)


def bar_rmse(pl, ncyc, outpath):
    plt = _plt()
    srcs = ["Model", "HRRR", "RTMA"]
    fig, axes = plt.subplots(1, NVAR, figsize=(11, 3.4), dpi=200)
    for i, v in enumerate(VARS):
        ax = axes[i]
        vals = [pl[v][s][1] for s in srcs]
        _annot(ax, ax.bar(range(len(srcs)), vals, 0.62, color=[C[s] for s in srcs], zorder=3))
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels(srcs, color=INK, fontsize=8)
        ax.set_title(f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
        _style(ax)
    axes[0].set_ylabel("RMSE (base units)", color=INK, fontsize=10)
    fig.suptitle(f"RMSE vs held-out obs  (10% hold-out, {ncyc} cycles)",
                 color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, outpath)


def box_percycle(rec, metric, outpath, jitter=0.08):
    plt = _plt()
    fig, axes = plt.subplots(1, NVAR, figsize=(12.5, 3.7), dpi=200)
    rng = np.random.default_rng(0)
    for i, v in enumerate(VARS):
        ax = axes[i]
        per = per_cycle(rec, i, metric)
        srcs = [s for s in ("Model", "HRRR", "RTMA") if per[s].size]
        data = [per[s] for s in srcs]
        pos = np.arange(len(srcs))
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color=BASELINE),
                        capprops=dict(color=BASELINE), boxprops=dict(lw=0))
        for patch, s in zip(bp["boxes"], srcs):
            patch.set_facecolor(C[s]); patch.set_alpha(0.30)
        for j, s in enumerate(srcs):
            y = per[s]
            xj = np.full(y.shape, pos[j]) + rng.uniform(-jitter, jitter, y.shape)
            ax.scatter(xj, y, s=6, color=C[s], alpha=0.5, edgecolors="none", zorder=3)
        ax.set_xticks(pos); ax.set_xticklabels(srcs, color=INK, fontsize=8.5)
        ax.set_title(v if metric == "corr" else f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        _style(ax)
    lab = "per-cycle correlation" if metric == "corr" else "per-cycle RMSE"
    sfx = "" if metric == "corr" else " (base units)"
    axes[0].set_ylabel(f"{lab}{sfx}", color=INK, fontsize=10)
    ncyc = int(np.unique(rec["cycle"]).size)
    fig.suptitle(f"Per-cycle skill distribution vs held-out obs  "
                 f"(each dot = 1 hourly cycle, {ncyc} cycles)",
                 color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, outpath)


def scatter(rec, outpath, sample=40000):
    plt = _plt()
    fig, axes = plt.subplots(1, NVAR, figsize=(12, 3.3), dpi=200)
    rng = np.random.default_rng(0)
    for i, v in enumerate(VARS):
        ax = axes[i]
        m = np.flatnonzero(rec["var"] == i)
        if sample and m.size > sample:
            m = rng.choice(m, sample, replace=False)
        t, p = rec["truth"][m], rec["model"][m]
        ax.scatter(t, p, s=2, alpha=0.15, color=C["Model"], edgecolors="none", zorder=3)
        if t.size:
            lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], color=BASELINE, lw=1, zorder=2)
        ax.set_title(f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        ax.set_xlabel("held-out obs", color=MUTED, fontsize=8)
        _style(ax)
    axes[0].set_ylabel("model analysis", color=INK, fontsize=10)
    fig.suptitle("Model analysis vs held-out obs", color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, outpath)


def error_hist(rec, outpath, bins=80):
    plt = _plt()
    fig, axes = plt.subplots(1, NVAR, figsize=(12, 3.3), dpi=200)
    for i, v in enumerate(VARS):
        ax = axes[i]
        m = rec["var"] == i
        t = rec["truth"][m]
        em, eh = rec["model"][m] - t, rec["hrrr"][m] - t
        rng = (np.nanpercentile(np.concatenate([em, eh]), [1, 99]) if em.size else (0, 1))
        ax.hist(eh, bins=bins, range=rng, color=C["HRRR"], alpha=0.55, label="HRRR", zorder=3)
        ax.hist(em, bins=bins, range=rng, color=C["Model"], alpha=0.55, label="Model", zorder=3)
        ax.axvline(0, color=BASELINE, lw=1, zorder=2)
        ax.set_title(f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        ax.set_xlabel("error (pred - obs)", color=MUTED, fontsize=8)
        _style(ax)
    axes[0].set_ylabel("count", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out error distribution", color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, outpath)


def spatial_error(rec, outpath, var="t", gridsize=60):
    plt = _plt()
    vi = VARS.index(var)
    m = rec["var"] == vi
    err = np.abs(rec["model"][m] - rec["truth"][m])
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    hb = ax.hexbin(rec["lon"][m], rec["lat"][m], C=err,
                   reduce_C_function=np.mean, gridsize=gridsize, cmap="magma")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(f"mean |error|  ({UNIT[var]})", color=INK, fontsize=9)
    ax.set_title(f"Spatial model error -- {var}", color=INK, fontsize=11)
    ax.set_xlabel("lon", color=MUTED, fontsize=9)
    ax.set_ylabel("lat", color=MUTED, fontsize=9)
    _style(ax); ax.yaxis.grid(False)
    fig.tight_layout()
    return _save(fig, outpath)


def print_table(pl, ncyc):
    print(f"\n{ncyc} cycles\n")
    print(f"{'var':<5}{'N':>9}  {'corr M/H/R':>26}   {'RMSE M/H/R (base units)':>28}  unit")
    for v in VARS:
        (cm, rm, n), (ch, rh, _), (cr, rr, _) = pl[v]["Model"], pl[v]["HRRR"], pl[v]["RTMA"]
        print(f"{v:<5}{n:>9}  {cm:>8.4f}{ch:>9.4f}{cr:>9.4f}   "
              f"{rm:>9.4f}{rh:>9.4f}{rr:>9.4f}  {UNIT[v]}")


# --------------------------------------------------------------------------- #
# Cache (per-point .npz -- lets --from-cache restyle with numpy+matplotlib only)
# --------------------------------------------------------------------------- #
def save_cache(rec, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(path, **rec)
    print(f"wrote cache -> {path}  ({rec['truth'].size} points)")


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_plots(rec, outdir):
    pl = pooled(rec)
    ncyc = int(rec["n_cycles"]) if "n_cycles" in rec else int(np.unique(rec["cycle"]).size)
    print_table(pl, ncyc)
    outs = [
        bar_corr(pl, ncyc, os.path.join(outdir, "heldout_correlation.png")),
        bar_rmse(pl, ncyc, os.path.join(outdir, "heldout_rmse.png")),
        box_percycle(rec, "corr", os.path.join(outdir, "heldout_percycle_corr.png")),
        box_percycle(rec, "rmse", os.path.join(outdir, "heldout_percycle_rmse.png")),
        scatter(rec, os.path.join(outdir, "heldout_scatter.png")),
        error_hist(rec, os.path.join(outdir, "heldout_error_hist.png")),
        spatial_error(rec, os.path.join(outdir, "heldout_spatial_t.png"), "t"),
    ]
    print("wrote:\n  " + "\n  ".join(outs))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", help="dir of inference_parallel.py output .nc")
    ap.add_argument("--glob", default="*.nc", help="filename glob within --indir")
    ap.add_argument("--outdir", default="Plots", help="dir for the PNGs")
    ap.add_argument("--cache", default=None, help="write per-point records .npz here")
    ap.add_argument("--from-cache", default=None,
                    help="skip inference outputs; re-plot from this records .npz")
    args = ap.parse_args()

    if args.from_cache:
        rec = load_cache(args.from_cache)
    else:
        if not args.indir:
            ap.error("--indir is required unless --from-cache is used")
        rec = read_records(args.indir, args.glob)
        if args.cache:
            save_cache(rec, args.cache)

    make_plots(rec, args.outdir)


if __name__ == "__main__":
    main()
