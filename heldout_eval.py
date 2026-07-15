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

from utils.misc_functions import model_label

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
# Consensus QC. A held-out ob is flagged as bad when all THREE independent
# products (Model, HRRR, RTMA) agree with EACH OTHER to within QC_AGREE_FRAC *
# QC_DISAGREE, yet their consensus (mean) disagrees with the ob by more than
# QC_DISAGREE base units. Three independent fields cannot mutually agree and all
# be wrong in the same direction -- so when they do, the ob is the outlier, not
# the analysis. These points are a tiny fraction of the sample but dominate the
# error budget (mostly badly-sited/mis-encoded mesonet stations); scoring on
# them measures observation error, not model skill.
# --------------------------------------------------------------------------- #
QC_DISAGREE = {"t": 5.0, "q": 3.0, "u10": 8.0, "v10": 8.0}   # base units
QC_AGREE_FRAC = 0.4      # products must agree to within this * QC_DISAGREE

# Title tags appended to every figure's suptitle; set per-mode by make_plots.
# _QC_TAG says how the obs were filtered, _MODEL_TAG which checkpoint produced them,
# _NET_DESC which observing network(s) the held-out obs were drawn from.
_QC_TAG = ""
_MODEL_TAG = ""
_NET_DESC = "both Mesonet and METAR"

NET_DESC = {"all": "both Mesonet and METAR", "metar": "METAR", "mesonet": "Mesonet"}


def _tag():
    return f"{_QC_TAG}{_MODEL_TAG}"


# --------------------------------------------------------------------------- #
# Stage 1: read inference outputs -> per-held-out-point records (base units).
# --------------------------------------------------------------------------- #
def _time_from_name(path):
    """'.../2023-01-15_12.nc' -> numpy datetime64 analysis hour."""
    stem = os.path.basename(path)[:-3]           # 2023-01-15_12
    return np.datetime64(f"{stem[:10]}T{stem[11:13]}", "h")


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
    # `cycle` is the index into `files`, so keep a time lookup over ALL files -- it stays
    # aligned even when a cycle is skipped. Indexed as cycle_times[cycle].
    cycle_times = np.array([_time_from_name(p) for p in files], dtype="datetime64[h]")
    n_used = n_skipped = 0
    model = ""
    for cyc, path in enumerate(files):
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if not model:
                model = model_label(ds) or ""
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
    rec["cycle_times"] = cycle_times
    rec["model_label"] = np.array(model)          # carried into the cache
    print(f"read {n_used} cycles ({n_skipped} skipped), "
          f"{rec['truth'].size} held-out points, model={model or 'unknown'}", flush=True)
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


def qc_mask(rec):
    """Boolean array over rec rows: True = consensus-flagged bad ob (drop these).

    Per variable, flags a point when the three products mutually agree (tight
    spread) but their consensus is far from the ob. See QC_DISAGREE above.
    """
    bad = np.zeros(rec["truth"].shape, dtype=bool)
    for vi, v in enumerate(VARS):
        m = rec["var"] == vi
        if not m.any():
            continue
        mo, h, r, t = (rec["model"][m], rec["hrrr"][m], rec["rtma"][m], rec["truth"][m])
        thr = QC_DISAGREE[v]
        spread = np.maximum.reduce([np.abs(mo - h), np.abs(mo - r), np.abs(h - r)])
        cons = (mo + h + r) / 3.0
        flag = (spread < thr * QC_AGREE_FRAC) & (np.abs(cons - t) > thr)
        bad[np.flatnonzero(m)[flag]] = True
    return bad


# Keys that are per-held-out-point (parallel arrays); everything else is metadata.
_POINT_KEYS = ("cycle", "var", "truth", "model", "hrrr", "rtma",
               "lat", "lon", "obs_source")


def apply_qc(rec):
    """Return (rec_without_flagged, bad_mask). Metadata keys pass through."""
    bad = qc_mask(rec)
    keep = ~bad
    out = {k: (v[keep] if k in _POINT_KEYS else v) for k, v in rec.items()}
    return out, bad


# obs_source as written by the data prep: 1 = mesonet, 2 = METAR (0 = unset/fill).
# METAR is ~11% of held-out points but is the CLEAN network -- mesonet carries most of
# the bad obs that consensus QC removes, so METAR-only scores answer a different and
# sharper question: skill against obs we actually trust.
SOURCE_CODE = {"mesonet": 1, "metar": 2}


def filter_source(rec, source):
    """Keep only points from one observing network. 'all' is a no-op."""
    if source == "all":
        return rec
    keep = rec["obs_source"] == SOURCE_CODE[source]
    return {k: (v[keep] if k in _POINT_KEYS else v) for k, v in rec.items()}


def qc_report(rec):
    """Print, per variable, how many points are flagged and what share of the
    (Model) MSE they carry -- the reason QC matters."""
    bad = qc_mask(rec)
    print("\nConsensus QC (flagged = all 3 products agree but disagree with ob):")
    print(f"{'var':<5}{'N':>9}{'flagged':>9}{'%':>7}   {'MSE share flagged':>18}")
    for vi, v in enumerate(VARS):
        m = rec["var"] == vi
        n = int(m.sum())
        if n == 0:
            continue
        b = bad[m]
        e2 = (rec["model"][m] - rec["truth"][m]) ** 2
        share = 100.0 * e2[b].sum() / e2.sum() if e2.sum() else 0.0
        print(f"{v:<5}{n:>9}{int(b.sum()):>9}{100*b.mean():>7.2f}{share:>17.1f}%")
    return bad


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
    ax.set_title(f"Analysis skill at held-out stations  (10% hold-out, {ncyc} cycles){_tag()}",
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
    fig.suptitle(f"RMSE vs held-out obs  (10% hold-out, {ncyc} cycles){_tag()}",
                 color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, outpath)


def violin_percycle(rec, metric, outpath, jitter=0.06):
    plt = _plt()
    fig, axes = plt.subplots(1, NVAR, figsize=(12.5, 3.7), dpi=200)
    rng = np.random.default_rng(0)
    for i, v in enumerate(VARS):
        ax = axes[i]
        per = per_cycle(rec, i, metric)
        srcs = [s for s in ("Model", "HRRR", "RTMA") if per[s].size]
        data = [per[s] for s in srcs]
        pos = np.arange(len(srcs))
        vp = ax.violinplot(data, positions=pos, widths=0.72,
                           showmeans=False, showmedians=False, showextrema=False)
        for body, s, y in zip(vp["bodies"], srcs, data):
            # The gaussian KDE smears mass past the data -- above corr=1, below rmse=0.
            # Clip each violin to the support actually observed.
            verts = body.get_paths()[0].vertices
            verts[:, 1] = np.clip(verts[:, 1], y.min(), y.max())
            body.set_facecolor(C[s]); body.set_alpha(0.32); body.set_edgecolor(C[s])
            body.set_linewidth(0.8); body.set_zorder(2)
        # Quartile box + median + 1.5*IQR whiskers, drawn inside each violin.
        for j, (s, y) in enumerate(zip(srcs, data)):
            q1, med, q3 = np.percentile(y, [25, 50, 75])
            iqr = q3 - q1
            lo = y[y >= q1 - 1.5 * iqr].min()
            hi = y[y <= q3 + 1.5 * iqr].max()
            ax.vlines(pos[j], lo, hi, color=BASELINE, lw=1.0, zorder=4)
            ax.vlines(pos[j], q1, q3, color=INK, lw=5.0, zorder=5)
            ax.scatter(pos[j], med, s=16, color="white", edgecolors="none", zorder=6)
        for j, s in enumerate(srcs):
            y = per[s]
            xj = np.full(y.shape, pos[j]) + rng.uniform(-jitter, jitter, y.shape)
            ax.scatter(xj, y, s=4, color=C[s], alpha=0.35, edgecolors="none", zorder=3)
        ax.set_xticks(pos); ax.set_xticklabels(srcs, color=INK, fontsize=8.5)
        ax.set_title(v if metric == "corr" else f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        _style(ax)
    lab = "per-cycle correlation" if metric == "corr" else "per-cycle RMSE"
    sfx = "" if metric == "corr" else " (base units)"
    axes[0].set_ylabel(f"{lab}{sfx}", color=INK, fontsize=10)
    ncyc = int(np.unique(rec["cycle"]).size)
    fig.suptitle(f"Per-cycle skill distribution vs held-out obs  "
                 f"(each dot = 1 hourly cycle, {ncyc} cycles){_tag()}",
                 color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, outpath)


def _week_window(rec, start, days):
    """Shared setup for the hourly time-series plots: require cycle_times, clip to
    [start, start + days), warn on missing hours, and slice the per-point subset.
    Returns (ctimes, t0, t1, keep_cyc, sub)."""
    if "cycle_times" not in rec:
        raise SystemExit("records have no 'cycle_times' -- regenerate the cache "
                         "with a current heldout_eval.py (--indir ...).")
    ctimes = rec["cycle_times"].astype("datetime64[h]")
    t0 = np.datetime64(start, "h")
    t1 = t0 + np.timedelta64(24 * days, "h")
    keep_cyc = np.flatnonzero((ctimes >= t0) & (ctimes < t1))
    if keep_cyc.size == 0:
        raise SystemExit(f"no cycles in [{t0}, {t1})")
    expected = 24 * days
    if keep_cyc.size != expected:
        missing = sorted(set(np.arange(t0, t1)) - set(ctimes[keep_cyc]))
        print(f"  note: {keep_cyc.size}/{expected} hours present; missing "
              f"{[str(m) for m in missing]}")
    sel = np.isin(rec["cycle"], keep_cyc)
    sub = {k: rec[k][sel] for k in ("cycle", "var", "truth", "model", "hrrr", "rtma")}
    return ctimes, t0, t1, keep_cyc, sub


def timeseries_rmse(rec, outpath, start, days=7):
    """Hourly per-cycle RMSE (Model/HRRR/RTMA) over `days` starting at `start`.

    One dot per analysis hour; x is the analysis time. RTMA is the reference
    curve -- note it ASSIMILATED these obs, so it is an optimistic upper bound,
    not a like-for-like competitor.
    """
    plt = _plt()
    import matplotlib.dates as mdates

    ctimes, t0, t1, keep_cyc, sub = _week_window(rec, start, days)

    fig, axes = plt.subplots(NVAR, 1, figsize=(13, 10), dpi=200, sharex=True)
    for i, v in enumerate(VARS):
        ax = axes[i]
        per = per_cycle(sub, i, "rmse")
        # per_cycle orders values by sorted unique cycle id within this var.
        cyc_ids = np.unique(sub["cycle"][sub["var"] == i])
        x = ctimes[cyc_ids].astype("datetime64[s]").astype(object)
        for s in ("HRRR", "RTMA", "Model"):
            y = per[s]
            if y.size != len(x):
                continue
            ax.plot(x, y, "-", color=C[s], lw=0.9, alpha=0.55, zorder=2)
            ax.plot(x, y, ".", color=C[s], ms=4.5, label=s, zorder=3)
        ax.set_ylabel(f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        _style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.5, alpha=0.6)

    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    ax = axes[-1]
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.set_xlabel("analysis time (UTC, one dot per hour)", color=INK, fontsize=10)
    fig.suptitle(f"Hourly RMSE vs held-out obs  ({str(t0)[:10]} to "
                 f"{str(t1 - np.timedelta64(1,'h'))[:10]}, {keep_cyc.size} cycles){_tag()}\n"
                 f"Analysis done on the 10% Held Out Observations of {_NET_DESC}",
                 color=INK, fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, outpath)


def timeseries_gap(rec, outpath, start, days=7):
    """Hourly RMSE(Model) - RMSE(RTMA), one panel per variable, in native units.

    Zero is parity with RTMA. Positive = the model trails RTMA that hour. RTMA
    assimilated these obs, so parity is already a strong result, not the target.
    """
    plt = _plt()
    import matplotlib.dates as mdates

    ctimes, t0, t1, keep_cyc, sub = _week_window(rec, start, days)

    fig, axes = plt.subplots(NVAR, 1, figsize=(13, 10), dpi=200, sharex=True)
    for i, v in enumerate(VARS):
        ax = axes[i]
        per = per_cycle(sub, i, "rmse")
        cyc_ids = np.unique(sub["cycle"][sub["var"] == i])
        x = ctimes[cyc_ids].astype("datetime64[s]").astype(object)
        # Guard the per-source alignment before differencing (matches timeseries_rmse).
        if per["Model"].size != len(x) or per["RTMA"].size != len(x):
            continue
        gap = per["Model"] - per["RTMA"]
        ax.axhline(0.0, color=BASELINE, lw=1.2, zorder=1)
        ax.plot(x, gap, "-", color=C["Model"], lw=0.9, alpha=0.55, zorder=2)
        # Colour each hour by who won it: blue = model beats RTMA, orange = RTMA wins.
        win = gap <= 0
        ax.plot(np.asarray(x)[win], gap[win], ".", color=C["Model"], ms=4.5, zorder=3)
        ax.plot(np.asarray(x)[~win], gap[~win], ".", color=C["RTMA"], ms=4.5, zorder=3)
        mean = gap.mean()
        ax.axhline(mean, color=C["Model"], lw=0.9, ls="--", alpha=0.7, zorder=2)
        ax.set_ylabel(f"{v}  ({UNIT[v]})", color=INK, fontsize=10)
        ax.annotate(f"mean {mean:+.3f}   model beats RTMA {100*win.mean():.0f}% of hours",
                    xy=(0.995, 0.92), xycoords="axes fraction", ha="right",
                    fontsize=8.5, color=MUTED)
        _style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.5, alpha=0.6)

    ax = axes[-1]
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.set_xlabel("analysis time (UTC, one dot per hour)", color=INK, fontsize=10)
    fig.suptitle(f"Model minus RTMA, hourly RMSE vs held-out obs  "
                 f"({str(t0)[:10]} to {str(t1 - np.timedelta64(1,'h'))[:10]}, "
                 f"{keep_cyc.size} cycles){_tag()}\n"
                 "0 = parity; above 0 the model trails RTMA (which assimilated these obs)",
                 color=INK, fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
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
    fig.suptitle(f"Model analysis vs held-out obs{_tag()}", color=INK, fontsize=11, y=1.02)
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
    fig.suptitle(f"Held-out error distribution{_tag()}", color=INK, fontsize=11, y=1.02)
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


def make_plots(rec, outdir, week_start=None, week_days=7, suffix="", tag="", model=None,
               source="all"):
    """Render the full figure set. `suffix` is inserted before '.png' in every
    filename (e.g. '_qc'); `tag` is appended to every figure title, as is the
    `model` label (e.g. 'e615'). `source` names the network(s) the obs came from."""
    global _QC_TAG, _MODEL_TAG, _NET_DESC
    _QC_TAG = tag
    _MODEL_TAG = f"  ·  {model}" if model else ""
    _NET_DESC = NET_DESC[source]

    def p(name):                                    # heldout_rmse.png -> heldout_rmse_qc.png
        base, ext = os.path.splitext(name)
        return os.path.join(outdir, f"{base}{suffix}{ext}")

    pl = pooled(rec)
    ncyc = int(rec["n_cycles"]) if "n_cycles" in rec else int(np.unique(rec["cycle"]).size)
    print_table(pl, ncyc)
    outs = [
        bar_corr(pl, ncyc, p("heldout_correlation.png")),
        bar_rmse(pl, ncyc, p("heldout_rmse.png")),
        violin_percycle(rec, "corr", p("heldout_percycle_corr.png")),
        violin_percycle(rec, "rmse", p("heldout_percycle_rmse.png")),
        scatter(rec, p("heldout_scatter.png")),
        error_hist(rec, p("heldout_error_hist.png")),
    ]
    if week_start and "cycle_times" in rec:
        outs.append(timeseries_rmse(
            rec, p("heldout_timeseries_rmse.png"), week_start, week_days))
        outs.append(timeseries_gap(
            rec, p("heldout_timeseries_gap.png"), week_start, week_days))
    print("wrote:\n  " + "\n  ".join(outs))
    _QC_TAG = _MODEL_TAG = ""
    _NET_DESC = NET_DESC["all"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", help="dir of inference_parallel.py output .nc")
    ap.add_argument("--glob", default="*.nc", help="filename glob within --indir")
    ap.add_argument("--outdir", default="Plots", help="dir for the PNGs")
    ap.add_argument("--cache", default=None, help="write per-point records .npz here")
    ap.add_argument("--from-cache", default=None,
                    help="skip inference outputs; re-plot from this records .npz")
    ap.add_argument("--week-start", default="2023-01-08",
                    help="first analysis day of the hourly RMSE time series "
                         "(YYYY-MM-DD); '' disables that plot")
    ap.add_argument("--week-days", type=int, default=7,
                    help="length of the hourly RMSE time series, in days")
    ap.add_argument("--qc", choices=("off", "on", "both"), default="off",
                    help="consensus QC of held-out obs. off = score all obs "
                         "(default, filenames unchanged); on = drop flagged obs "
                         "and write '_qc' figures only; both = write both sets.")
    ap.add_argument("--label", default=None,
                    help="model label for the figure titles, e.g. e615 "
                         "(default: the checkpoint epoch recorded in the inputs)")
    ap.add_argument("--source", choices=("all", "metar", "mesonet"), default="all",
                    help="score only one observing network. all = both (default, "
                         "filenames unchanged); metar/mesonet write '_metar'/'_mesonet' "
                         "figures. METAR is the clean network (~11%% of held-out points).")
    args = ap.parse_args()

    if args.from_cache:
        rec = load_cache(args.from_cache)
    else:
        if not args.indir:
            ap.error("--indir is required unless --from-cache is used")
        rec = read_records(args.indir, args.glob)
        if args.cache:
            save_cache(rec, args.cache)

    week = args.week_start or None
    model = args.label or str(rec.get("model_label", "")) or None

    n_all = rec["truth"].size
    rec = filter_source(rec, args.source)
    src_sfx = "" if args.source == "all" else f"_{args.source}"
    src_tag = "" if args.source == "all" else f"  [{args.source.upper()} only]"
    if args.source != "all":
        print(f"\n{args.source.upper()} only: {rec['truth'].size:,} of {n_all:,} "
              f"held-out points ({100*rec['truth'].size/max(n_all,1):.1f}%)")
        if rec["truth"].size == 0:
            raise SystemExit(f"no {args.source} points in this record set")

    # Always report what QC would remove, so the trade-off is visible even in 'off'.
    qc_report(rec)

    if args.qc in ("off", "both"):
        print("\n===== ALL held-out obs (no QC) =====")
        make_plots(rec, args.outdir, week, args.week_days,
                   suffix=src_sfx, tag=src_tag, model=model, source=args.source)

    if args.qc in ("on", "both"):
        rec_qc, bad = apply_qc(rec)
        n0, n1 = bad.size, int((~bad).sum())
        print(f"\n===== QC'd held-out obs ({n0 - n1} of {n0} flagged and removed, "
              f"{100*(n0 - n1)/max(n0,1):.2f}%) =====")
        make_plots(rec_qc, args.outdir, week, args.week_days,
                   suffix=f"{src_sfx}_qc",
                   tag=f"{src_tag}  [consensus-QC: flagged obs removed]", model=model,
                   source=args.source)


if __name__ == "__main__":
    main()
