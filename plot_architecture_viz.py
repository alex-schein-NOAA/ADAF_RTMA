#!/usr/bin/env python
"""Architecture-explainer figures for the lowres_r4_metar_l0p3 e155 checkpoint.

Produces the "what is the model doing" figure set (high-signal core):
  1a  conv_first as an obs-minus-HRRR innovation detector
  1b  layer_scale.gamma -- the body's fade-in from its 1e-4 init
  1c  Swin relative-position bias -- the learned attention prior (8 blocks x 6 heads)
  2c  innovation vs. model correction on a real cycle (from existing inference output)

All four are CPU-only. The single-ob point-spread figure (3a) needs a forward pass and
lives in plot_impulse_response.py (GPU sbatch).

Run:
  python plot_architecture_viz.py            # all four
  python plot_architecture_viz.py --figs 1a 2c
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "ckpt_snapshots/lowres_r4_metar_l0p3_e155_17346342.tar"
INFER_NC = "inference_out/jan2023_r4_l0p3_e155/2023-01-01_18.nc"
OUTDIR = "Plots/jan2023_r4_l0p3_e155/architecture"

# conv_first input-channel layout (in_chans = 17), from inference_parallel.build_model_input:
#   0-3   hrrr_q, hrrr_t, hrrr_u10, hrrr_v10
#   4-6   sta_q  x 3 obs-time windows (oldest -> analysis)
#   7-9   sta_t  x 3
#   10-12 sta_u10 x 3
#   13-15 sta_v10 x 3
#   16    z (topography)
CH_LABELS = (
    ["hrrr_q", "hrrr_t", "hrrr_u10", "hrrr_v10"]
    + [f"sta_q[{w}]" for w in range(3)]
    + [f"sta_t[{w}]" for w in range(3)]
    + [f"sta_u10[{w}]" for w in range(3)]
    + [f"sta_v10[{w}]" for w in range(3)]
    + ["topo"]
)
# stats.csv min-max ranges for physical <-> normalized conversion
RANGE = {"q": 0.025, "t": 90.0, "u10": 50.0, "v10": 50.0}  # vmax - vmin
VMIN = {"q": 0.0, "t": -40.0, "u10": -25.0, "v10": -25.0}


def load_state():
    ck = torch.load(CKPT, map_location="cpu")
    sd = {}
    for k, v in ck["model_state"].items():
        for pre in ("_orig_mod.", "module."):
            k = k.replace(pre, "")
        sd[k] = v
    return sd, ck.get("epoch"), ck.get("iters")


# ---------------------------------------------------------------- 1a
def fig_1a(sd):
    w = sd["conv_first.weight"].float().numpy()  # (96, 17, 3, 3)
    # total signed gain of each output filter on each input channel (sum over the 3x3)
    gain = w.sum(axis=(2, 3))  # (96, 17)

    # order the 96 filters by their temperature-innovation score so the pattern is legible:
    # a filter that computes (sta_t - hrrr_t) has +gain on sta_t and -gain on hrrr_t.
    hrrr_t = gain[:, 1]
    sta_t = gain[:, 7:10].sum(axis=1)  # all three obs windows
    innov_t = sta_t - hrrr_t
    order = np.argsort(innov_t)

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.28)

    # left: 96x17 channel-attribution heatmap
    ax = fig.add_subplot(gs[0])
    vmax = np.percentile(np.abs(gain), 99)
    im = ax.imshow(gain[order].T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(17))
    ax.set_yticklabels(CH_LABELS, fontsize=8)
    ax.set_xlabel("output filter (of 96), sorted by temperature-innovation score")
    ax.set_title("conv_first: signed gain of each filter on each input channel\n"
                 "(red = amplifies, blue = subtracts)")
    # bracket the hrrr_t and sta_t rows
    for row, c in [(1, "k"), (7, "k"), (8, "k"), (9, "k")]:
        ax.axhline(row - 0.5, color=c, lw=0.3, alpha=0.3)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="summed 3x3 weight")

    # right: the money scatter -- hrrr_t gain vs sta_t gain, per filter
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.axvline(0, color="0.7", lw=0.6)
    lim = max(np.abs(hrrr_t).max(), np.abs(sta_t).max()) * 1.05
    ax2.plot([-lim, lim], [lim, -lim], "--", color="tab:green", lw=1.2,
             label="obs - HRRR axis\n(perfect innovation)")
    sc = ax2.scatter(hrrr_t, sta_t, c=innov_t, cmap="RdBu_r",
                     vmin=-np.abs(innov_t).max(), vmax=np.abs(innov_t).max(),
                     s=22, edgecolor="0.3", linewidth=0.3)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("gain on hrrr_t")
    ax2.set_ylabel("gain on sta_t (all windows)")
    ax2.set_title("each dot = one filter\nspread vertically (obs), clustered at x~0 (HRRR)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_aspect("equal")

    obs_g = np.abs(gain[:, 4:16]).mean()   # mean |gain| on the 12 obs channels
    hrrr_g = np.abs(gain[:, 0:4]).mean()   # mean |gain| on the 4 HRRR channels
    fig.suptitle("1a  conv_first is an observation encoder   |   "
                 f"mean |gain| on obs channels is {obs_g/hrrr_g:.1f}x that on HRRR "
                 f"(strongest on analysis-time sta_t[2]); the HRRR subtraction is completed downstream",
                 fontsize=12, y=1.0)
    _save(fig, "1a_conv_first_innovation.png")


# ---------------------------------------------------------------- 1d / 1dd / 1e / 1ee
#
# The "temperature-innovation score" that orders the columns:
#   score(filter) = gain(sta_t) - gain(hrrr_t),  gain = sum of that filter's 3x3 weights
#     on the given input channel (sta_t summed over all 3 obs-time windows).
# A filter that literally computed the analysis increment sta_t - hrrr_t would put +weight
# on the observed temperature and -weight on the background temperature, giving a large
# POSITIVE score. Sorting ascending puts the strongest "background-minus-obs" filters on the
# far left and the strongest "obs-minus-background" (true innovation) filters on the far
# right, so the temperature-differencing structure -- if conv_first did any -- would show as
# a left-to-right gradient. (1a's finding: it mostly doesn't; the subtraction is downstream.)
#
# TWO kernel-character metrics are plotted, as a matched pair (1e/1ee/1f use "dc",
# 1eR/1eeR/1fR use "rough"). They answer DIFFERENT questions -- see the table below.
#
#   "dc"     edge-ness         1 - |Σw| / Σ|w|              in [0, 1]
#            "does this kernel pass DC, or cancel it?"  0 = the weights all share a sign
#            (a level/smoothing filter), 1 = they sum to zero (a derivative filter).
#            NOT a published metric -- an ad-hoc ratio local to this repo -- and it is
#            PERMUTATION-INVARIANT: it cannot see where in the 3x3 the weights sit, so a
#            checkerboard and a scrambled checkerboard score identically.
#
#   "rough"  Laplacian roughness  R = Σ(Δw)² / Σw²          in [0, 6], plotted as R/6
#            "is this kernel spatially wiggly?"  Sum of squared differences between
#            4-neighbour adjacent weights, over the sum of squared weights. This is the
#            Rayleigh quotient wᵀLw/wᵀw of the 3x3 grid-graph Laplacian, a.k.a. the
#            normalized Dirichlet energy -- standard in graph signal processing. Scale
#            invariant, exactly 0 for a constant kernel, and 6 is the true max (largest
#            eigenvalue of the 4-neighbour 3x3 grid Laplacian), so R/6 lands in [0, 1].
#
# They correlate but are NOT interchangeable (Spearman 0.44-0.79 on these weights):
#
#     kernel                    edge-ness    R/6
#     box blur (all +1/9)          0.00      0.00
#     gaussian [1,2,1]^2           0.00      0.11
#     ramp [-1, 0, +1]             1.00      0.17   <- zero-sum but spatially smooth
#     sobel x                      1.00      0.22
#     delta (centre spike)         0.00      0.67   <- passes DC but is maximally peaky
#     laplacian (4-nbr)            1.00      0.90
#
# The two rows marked above are each metric's blind spot. Keep both: the "obs enter as
# levels, HRRR enters as gradients" finding is a claim about DC gain, which is what
# edge-ness measures; roughness is the better answer to the literal word "edge".

LAPL_MAX = 6.0          # largest eigenvalue of the 4-neighbour 3x3 grid-graph Laplacian


def _edgeness(w):
    """Per-kernel edge-ness 1-|Σw|/Σ|w| for (...,3,3) weights. In [0,1]."""
    l1 = np.abs(w).sum(axis=(-2, -1))
    return 1.0 - np.abs(w.sum(axis=(-2, -1))) / (l1 + 1e-9)


def _roughness(w, normalize=True):
    """Laplacian roughness R = Σ(Δw)²/Σw² over 4-neighbour adjacent pairs, for (...,3,3)
    weights. Raw R is in [0, LAPL_MAX]; normalize=True returns R/LAPL_MAX in [0,1]."""
    dh = np.diff(w, axis=-1)                               # horizontal neighbour diffs
    dv = np.diff(w, axis=-2)                               # vertical neighbour diffs
    num = (dh ** 2).sum(axis=(-2, -1)) + (dv ** 2).sum(axis=(-2, -1))
    r = num / ((w ** 2).sum(axis=(-2, -1)) + 1e-12)
    return r / LAPL_MAX if normalize else r


# per-metric plot copy, so the paired figures label themselves correctly
METRIC = {
    "dc": dict(
        fn=_edgeness,
        axis="edge-ness   1 − |Σw| / Σ|w|",
        short="edge-ness",
        hi="edge", lo="smoother",
        frame="edge-ness",
        # edge-ness genuinely uses its full [0,1] range (it saturates at both ends)
        vmax=1.0, cticks=[0.5], ctlabels=["0.5"],
    ),
    "rough": dict(
        fn=lambda w: _roughness(w, normalize=True),
        axis="Laplacian roughness   R = Σ(Δw)² / Σw²      (÷6)",
        short="Laplacian roughness R",
        hi="rough", lo="smooth",
        frame="Laplacian roughness R",
        # R/6 only reaches ~0.86 here and 99% of kernels sit under 0.72, so a [0,1] frame
        # scale wastes most of the colormap and everything reads as flat purple. Clip the
        # frames at 0.6 (>= that saturates yellow) to spend the ramp where the data is.
        # The 1fR histogram deliberately keeps the full [0,1] so its canonical-kernel
        # reference marks (delta 0.67, laplacian 0.90) stay on scale.
        # no 0.0 tick: the "smooth" legend text sits at the bar's foot and would collide
        vmax=0.6, cticks=[0.3, 0.6], ctlabels=["0.3", "≥0.6"],
    ),
}

# canonical 3x3 kernels, as reference marks on the roughness axis (R/6)
REF_KERNELS = [(0.000, "box"), (0.111, "gauss"), (0.222, "sobel"),
               (0.667, "delta"), (0.900, "lapl")]


def _kernel_grid(w, pad, keep=None, metric="dc"):
    """Lay out conv_first's (96,17,3,3) weights as a 17-row x N-col mosaic of the actual
    3x3 kernels, columns in temperature-innovation order (same sort as 1a).

    pad    -- grout cells around each 3x3 tile. 0 = tiles butt together (no white space);
              >=1 leaves a frame that _kernel_grid fills with the tile's METRIC score.
    keep   -- if given, keep only the `keep` leftmost + `keep` rightmost columns and drop the
              middle (for a taller ~2:1 crop); returns the cut position in tiles.
    metric -- "dc" (edge-ness) or "rough" (Laplacian roughness R/6); both land in [0,1] so
              the frames share one viridis scale and the pair reads side by side.

    Returns dict with kern, edge canvases, cell, score, ncol, cut.
    """
    hrrr_t = w[:, 1].sum(axis=(1, 2))
    sta_t = w[:, 7:10].sum(axis=(1, 2, 3))
    order = np.argsort(sta_t - hrrr_t)                     # ascending innovation score

    if keep is not None:
        cols = np.concatenate([order[:keep], order[-keep:]])
        cut = keep                                        # tiles before the omission gap
    else:
        cols = order
        cut = None

    score = METRIC[metric]["fn"](w)                        # (96,17), in [0,1]

    k = 3
    cell = k + 2 * pad
    nrow, ncol = 17, len(cols)
    kern = np.full((nrow * cell, ncol * cell), np.nan)
    edge = np.full((nrow * cell, ncol * cell), np.nan)
    for ri in range(nrow):                                # input channel -> row
        for cj, oc in enumerate(cols):                    # sorted output filter -> col
            r0, c0 = ri * cell, cj * cell
            if pad:
                edge[r0:r0 + cell, c0:c0 + cell] = score[oc, ri]
            kern[r0 + pad:r0 + pad + k, c0 + pad:c0 + pad + k] = w[oc, ri]
    return dict(kern=kern, edge=edge, cell=cell, score=score, ncol=ncol, cut=cut)


def _render_kernel_grid(sd, pad, keep, framed, tag, name, metric="dc"):
    m = METRIC[metric]
    w = sd["conv_first.weight"].float().numpy()           # (96,17,3,3)
    g = _kernel_grid(w, pad, keep, metric)
    kern, edge, cell, ncol, cut = g["kern"], g["edge"], g["cell"], g["ncol"], g["cut"]
    nrow = 17
    vext = np.percentile(np.abs(w), 99)
    mean_score = g["score"].mean()

    # size the figure to the data so equal-area cells leave no interior white space.
    data_aspect = ncol / nrow                             # width/height in cells
    if framed:
        # manual layout: the mosaic rectangle is sized directly to the data (aspect="auto"
        # fills it, so cells stay square and the whole figure lands near the mosaic's own
        # aspect -- ~2:1 for the crop). Both legends are compressed into one thin right strip.
        mos_h = 6.0 if keep is not None else 3.3          # inches
        mos_w = mos_h * data_aspect
        Lm, Rm, Tm, Bm = 1.0, 1.1, 0.5, 0.62              # inch margins
        figw, figh = Lm + mos_w + Rm, Tm + mos_h + Bm
        fig = plt.figure(figsize=(figw, figh))
        ax = fig.add_axes([Lm / figw, Bm / figh, mos_w / figw, mos_h / figh])
        ime = ax.imshow(edge, cmap="viridis", vmin=0.0, vmax=m["vmax"], aspect="auto",
                        interpolation="nearest")
        imk = ax.imshow(kern, cmap="RdBu_r", vmin=-vext, vmax=vext, aspect="auto",
                        interpolation="nearest")
    else:
        box = nrow / ncol                                 # 0.177 full, 0.50 for the 2:1 crop
        figw = 18 if keep is None else 12
        figh = figw * box + 1.9
        fig, ax = plt.subplots(figsize=(figw, figh), layout="constrained")
        imk = ax.imshow(kern, cmap="RdBu_r", vmin=-vext, vmax=vext, aspect="auto",
                        interpolation="nearest")
        ax.set_box_aspect(box)

    ax.set_yticks([ri * cell + cell / 2 - 0.5 for ri in range(nrow)])
    ax.set_yticklabels(CH_LABELS, fontsize=8)
    ax.set_xticks([])
    for ri in range(1, nrow):                             # faint row separators only
        ax.axhline(ri * cell - 0.5, color="0.55", lw=0.4)

    if cut is not None:                                   # mark the omitted middle
        xd = cut * cell - 0.5
        ax.axvline(xd, color="k", lw=2.0)
        ax.text(xd, nrow * cell / 2, f"middle {96 - 2 * keep} filters omitted",
                ha="center", va="center", rotation=90, fontsize=8, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", lw=0.5))
        xlab = (f"{ncol} of 96 output filters: {keep} strongest background-minus-obs "
                f"(left)  |  {keep} strongest obs-minus-background (right)")
    else:
        xlab = ("output filter (of 96), sorted by temperature-innovation score  "
                "(left = background-minus-obs, right = obs-minus-background)")
    ax.set_xlabel(xlab, fontsize=9)

    if framed:
        if metric == "dc":
            frac_edge = (g["score"] > 0.5).mean()
            note = f"{frac_edge*100:.0f}% of {nrow*ncol} shown are edge/gradient, >0.5"
        else:
            # a >0.5 cut is meaningless on R/6 (only ~16% clear it) -- quote the mean,
            # and be explicit that the frame ramp is clipped rather than full-range
            clipped = (g["score"] > m["vmax"]).mean()
            note = (f"mean R/6 = {mean_score:.2f} of {nrow*ncol}; ramp clipped at "
                    f"{m['vmax']:g}, top {clipped*100:.0f}% saturated")
        ax.set_title(f"{tag}  conv_first 3x3 kernels, each framed by its "
                     f"{m['frame']}   ({note})", fontsize=10)
        # compress both legends into one thin right-hand strip: the metric bar on top
        # (hi label over it, lo label below), the weight-value bar stacked beneath it.
        sx, sw = (Lm + mos_w + 0.12) / figw, 0.16 / figw
        gap = 0.16 * mos_h
        h_each = (mos_h - gap) / 2
        cax_e = fig.add_axes([sx, (Bm + gap + h_each) / figh, sw, h_each / figh])
        cax_w = fig.add_axes([sx, Bm / figh, sw, h_each / figh])
        cbe = fig.colorbar(ime, cax=cax_e)
        cbe.set_ticks(m["cticks"]); cbe.set_ticklabels(m["ctlabels"])
        cax_e.text(0.5, 1.03, m["hi"], transform=cax_e.transAxes, ha="center", va="bottom",
                   fontsize=9)
        # anchor the low label to the RIGHT of the thin bar so it flows into the clear
        # margin instead of spilling left over the mosaic
        cax_e.text(1.6, 0.0, m["lo"], transform=cax_e.transAxes, ha="left", va="center",
                   fontsize=9)
        cbw = fig.colorbar(imk, cax=cax_w)
        cbw.set_label("weight", fontsize=9)
    else:
        ax.set_title(f"{tag}  conv_first: every 3x3 kernel  ({nrow} input channels x {ncol} "
                     f"output filters = {nrow*ncol} kernels; red +, blue -)", fontsize=11)
        fig.colorbar(imk, ax=ax, fraction=0.02, pad=0.01, label="weight value")

    _save(fig, name, tight=False)


def fig_1d(sd):
    """Like 1a, but every cell is the actual 3x3 kernel, tiles butted (no white space)."""
    _render_kernel_grid(sd, pad=0, keep=None, framed=False, tag="1d",
                        name="1d_conv_first_kernels.png")


def fig_1dd(sd):
    """1d cropped to the extremes (2:1) -- strongest innovation filters, middle dropped."""
    _render_kernel_grid(sd, pad=0, keep=17, framed=False, tag="1dd",
                        name="1dd_conv_first_kernels_2to1.png")


def fig_1e(sd):
    """1d, plus each kernel framed by its edge-ness (smoother <-> edge) with a legend."""
    _render_kernel_grid(sd, pad=1, keep=None, framed=True, tag="1e",
                        name="1e_conv_first_kernels_edgeness.png", metric="dc")


def fig_1ee(sd):
    """1e cropped to the extremes (2:1) -- framed by edge-ness, middle dropped."""
    _render_kernel_grid(sd, pad=1, keep=17, framed=True, tag="1ee",
                        name="1ee_conv_first_kernels_edgeness_2to1.png", metric="dc")


def fig_1eR(sd):
    """1e's twin, framed by Laplacian roughness R/6 instead of edge-ness."""
    _render_kernel_grid(sd, pad=1, keep=None, framed=True, tag="1eR",
                        name="1eR_conv_first_kernels_roughness.png", metric="rough")


def fig_1eeR(sd):
    """1ee's twin, framed by Laplacian roughness R/6 instead of edge-ness."""
    _render_kernel_grid(sd, pad=1, keep=17, framed=True, tag="1eeR",
                        name="1eeR_conv_first_kernels_roughness_2to1.png", metric="rough")


# ---------------------------------------------------------------- 1f
# edge-ness palette matched to plot_architecture_viz2.py's 4d:
# purple = level/smoother, amber = gradient/edge (as in 1e's viridis frames).
SMOOTH_C = "#4c2a85"
EDGE_C = "#e0a800"
EDGE_TXT = "#9a7500"


def fig_1f(sd):
    """1f -- just conv_first's edge-ness (the 96x17 kernel pairs), one clean histogram."""
    w = sd["conv_first.weight"].float().numpy()            # (96, 17, 3, 3)
    e = _edgeness(w).ravel()

    counts, edges = np.histogram(e, bins=44, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = np.where(centers > 0.5, EDGE_C, SMOOTH_C)

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=colors,
           edgecolor="white", lw=0.3, zorder=3)
    ax.axvline(0.5, color="k", ls="--", lw=1.8, zorder=4)
    frac_edge = float((e > 0.5).mean())

    ax.set_title(f"conv_first   Conv2d(17 → 96)\nfull res · {w.shape[0] * w.shape[1]:,} "
                 "kernels", fontsize=17, pad=12)
    ax.set_xlabel("edge-ness   1 − |Σw| / Σ|w|", fontsize=15)
    ax.set_ylabel("in-out kernel pairs", fontsize=15)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color="0.85", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.04, 0.95, f"{(1 - frac_edge) * 100:.0f}%\nsmoother", transform=ax.transAxes,
            ha="left", va="top", fontsize=15, color=SMOOTH_C, weight="bold")
    ax.text(0.96, 0.95, f"{frac_edge * 100:.0f}%\nedge", transform=ax.transAxes,
            ha="right", va="top", fontsize=15, color=EDGE_TXT, weight="bold")

    fig.suptitle("1f  conv_first edge-ness — gradient/edge (amber) vs level/smoothing "
                 "(purple)", fontsize=16, y=1.0)
    _save(fig, "1f_conv_first_edgeness.png")


def roughness_hist(ax, r, subtitle, bins=44):
    """1f's twin panel, on Laplacian roughness R/6.

    Deliberately NOT a two-sided threshold plot like the edge-ness histograms: a >0.5 cut
    is arbitrary here (only ~16% of conv_first's kernels clear it), so instead the bars
    carry the same viridis ramp used for 1e's frames and canonical kernels are marked on
    the axis, which lets a reader place the distribution without inventing a threshold.
    """
    counts, edges = np.histogram(r, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9,
           color=plt.get_cmap("viridis")(centers), edgecolor="white", lw=0.3, zorder=3)

    med, mean = float(np.median(r)), float(r.mean())
    ax.axvline(med, color="k", ls="--", lw=1.8, zorder=5)
    ax.text(med, 0.97, f"  median {med:.2f}", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=13, weight="bold")

    top = counts.max()
    for x, lab in REF_KERNELS:                             # canonical kernels for scale
        ax.plot([x, x], [-0.055 * top, -0.015 * top], color="0.35", lw=1.4,
                clip_on=False, zorder=6)
        ax.text(x, -0.075 * top, lab, ha="center", va="top", fontsize=9, color="0.35")

    ax.set_title(subtitle, fontsize=17, pad=12)
    ax.set_xlabel(METRIC["rough"]["axis"], fontsize=15, labelpad=26)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color="0.85", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return mean, med


def fig_1fR(sd):
    """1fR -- 1f's twin: conv_first's Laplacian roughness R, one clean histogram."""
    w = sd["conv_first.weight"].float().numpy()            # (96, 17, 3, 3)
    r = _roughness(w).ravel()

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    mean, _ = roughness_hist(ax, r, f"conv_first   Conv2d(17 → 96)\nfull res · "
                             f"{w.shape[0] * w.shape[1]:,} kernels")
    ax.set_ylabel("in-out kernel pairs", fontsize=15)

    fig.suptitle(f"1fR  conv_first Laplacian roughness R — spatial wiggliness "
                 f"(mean R/6 = {mean:.2f}; 1f's twin, DC gain → roughness)",
                 fontsize=15, y=1.0)
    _save(fig, "1fR_conv_first_roughness.png")


# ---------------------------------------------------------------- 1b
def fig_1b(sd):
    g = sd["layer_scale.gamma"].float().numpy()  # (192,)
    init = 1e-4
    order = np.argsort(g)
    gs = g[order]

    mabs = np.abs(g).mean()

    # One panel, slide-sized: the 192 gates, nothing else. (The old |gamma| histogram
    # second panel was dropped -- the growth-from-init story is already carried by the
    # +/-init reference lines here.)
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(gs))
    ax.bar(x, gs, width=1.0,
           color=np.where(gs >= 0, "tab:red", "tab:blue"), linewidth=0)

    ax.axhline(0, color="0.55", lw=1.0, zorder=1)
    for s in (+1, -1):
        ax.axhline(s * init, color="k", ls="--", lw=2.0, zorder=3,
                   label="init = $\\pm$1e-4 (indistinguishable from 0 at this scale)"
                   if s > 0 else None)
    for s in (+1, -1):
        ax.axhline(s * mabs, color="tab:orange", ls="-", lw=2.0, zorder=3,
                   label=f"mean |$\\gamma$| = {mabs:.1e}  ({mabs/init:.0f}$\\times$ init)"
                   if s > 0 else None)

    ax.set_xlim(-1, len(gs))
    ax.set_xticks(np.arange(0, len(gs) + 1, 16))
    ax.set_xlabel("body channel (of 192), sorted by $\\gamma$", fontsize=24, labelpad=10)
    ax.set_ylabel("layer_scale  $\\gamma$", fontsize=24, labelpad=10)
    ax.tick_params(labelsize=19, length=6, width=1.2)
    ax.legend(fontsize=19, loc="upper left", framealpha=0.95)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # the init line is invisibly thin next to the grown gates -- say so in words too
    ax.annotate(f"every gate initialized at +1e-4;\n"
                f"|$\\gamma$| now spans {np.abs(g).min():.1e} – {np.abs(g).max():.1e}",
                xy=(0.985, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=19, bbox=dict(boxstyle="round,pad=0.5", fc="0.95", ec="0.7"))

    fig.suptitle("1b  LayerScale: the per-channel gate on the attention body's output\n"
                 "opened ~50$\\times$ from init — the body earned its contribution "
                 "(the pyramid's was zeroed)", fontsize=25, y=0.995, va="top")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.13)
    _save(fig, "1b_layerscale_gamma.png", tight=False)


# ---------------------------------------------------------------- 1c
def fig_1c(sd):
    keys = sorted([k for k in sd if k.endswith("relative_position_bias_table")],
                  key=lambda k: (int(k.split("body.")[1].split(".")[0]),
                                 int(k.split("blocks.")[1].split(".")[0])))
    n_blocks = len(keys)                       # 8
    tbl0 = sd[keys[0]].float().numpy()
    n_heads = tbl0.shape[1]                    # 6
    w = int(round((tbl0.shape[0] ** 0.5)))     # 23 = 2*window-1
    win = (w + 1) // 2                          # 12

    # heads down the left (rows), blocks across the top (cols) -> landscape, square tiles
    fig, axes = plt.subplots(n_heads, n_blocks, figsize=(1.9 * n_blocks, 1.9 * n_heads))
    # shared symmetric color scale across all maps
    allb = np.concatenate([sd[k].float().numpy().ravel() for k in keys])
    vext = np.percentile(np.abs(allb), 99)

    for bi, k in enumerate(keys):
        tbl = sd[k].float().numpy()            # (529, n_heads)
        blk = int(k.split("body.")[1].split(".")[0])
        blkpos = int(k.split("blocks.")[1].split(".")[0])
        for hi in range(n_heads):
            ax = axes[hi, bi]
            m = tbl[:, hi].reshape(w, w)
            ax.imshow(m, cmap="RdBu_r", vmin=-vext, vmax=vext, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if hi == 0:
                ax.set_title(f"RSTB{blk}.b{blkpos}", fontsize=22)
            if bi == 0:
                ax.set_ylabel(f"head {hi}", fontsize=22)

    fig.suptitle("1c  Swin relative-position bias = the learned attention prior\n"
                 f"each tile is a {w}x{w} map over relative offsets "
                 f"(window {win} @ 1/4 res -> {win*4} grid cells reach); "
                 "red = attends toward, blue = away", fontsize=20, y=0.995,
                 va="top")
    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.01,
                        wspace=0.06, hspace=0.06)
    _save(fig, "1c_swin_relpos_bias.png", tight=False)


# ---------------------------------------------------------------- 2c
def fig_2c():
    import xarray as xr
    ds = xr.open_dataset(INFER_NC)
    var = "t"  # temperature is the most legible

    # All fields in the inference output are already in physical units.
    hrrr_p = ds[f"hrrr_{var}"].to_numpy()
    sta = ds[f"sta_{var}"]
    sta_p = sta.isel(obs_time_window=-1).to_numpy() if "obs_time_window" in sta.dims else sta.to_numpy()
    resid_p = ds[f"output_residual_{var}"].to_numpy()

    # Stations = METAR only (obs_source == 2). The model was trained train_obs_source: metar,
    # so it never saw a mesonet ob -- restrict every station overlay to METAR.
    obs_source = ds["obs_source"].to_numpy()
    heldout = ds["heldout_mask"].to_numpy() > 0
    station = obs_source == 2                         # METAR cells
    ys, xs = np.where(station)
    innov = sta_p - hrrr_p                            # obs - HRRR at METAR cells

    # Geographic coords. NOTE row 0 is SOUTH (lat increases with row) -> origin="lower"
    # everywhere so the maps are north-up.
    lat = ds["lat"].to_numpy()
    lon = ds["lon"].to_numpy()

    rext = np.nanpercentile(np.abs(resid_p), 99.5)
    iext = np.nanpercentile(np.abs(innov[station]), 98)
    cyc = os.path.basename(INFER_NC)

    # The two zoom panels differ only in their reference (HRRR vs RTMA), so they get ONE
    # shared symmetric K scale -- otherwise the model's small departure from RTMA is
    # auto-stretched and looks as large as its innovation over HRRR.
    rtma_p = ds[f"rtma_{var}"].to_numpy()
    dmod = ds[f"output_{var}"].to_numpy() - rtma_p     # model - RTMA (full grid)
    dobs = sta_p - rtma_p                              # obs   - RTMA (at METAR cells)
    zext = max(rext, iext,
               np.nanpercentile(np.abs(dmod), 99.5),
               np.nanpercentile(np.abs(dobs[station]), 98))

    # zoom box: Canada just north of Montana (lat 49-52 N, lon -114..-103 = 246-257 E)
    box = (lat >= 49.0) & (lat <= 52.3) & (lon >= 246.0) & (lon <= 257.0)
    byx = np.where(box)
    y0, y1 = byx[0].min(), byx[0].max() + 1
    x0, x1 = byx[1].min(), byx[1].max() + 1

    from matplotlib.patches import Rectangle

    def zoom_box(ax):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="k", lw=2.0))

    # ---- (i) METAR difference from HRRR ----
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(hrrr_p, cmap="Greys", alpha=0.20, origin="lower")
    sc = ax.scatter(xs, ys, c=innov[ys, xs], cmap="RdBu_r", vmin=-iext, vmax=iext,
                    s=11, linewidths=0)
    zoom_box(ax)
    ax.set_title(f"2c(i)  METAR HRRR difference  obs - HRRR  [K]   |   {cyc}   |   "
                 f"{station.sum()} METAR stations", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="obs - HRRR [K]")
    _save(fig, "2c_i_innovation.png")

    # ---- (ii) model innovation ----
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(resid_p, cmap="RdBu_r", vmin=-rext, vmax=rext, origin="lower")
    zoom_box(ax)
    ax.set_title(f"2c(ii)  model innovation  output - HRRR  [K]   |   {cyc}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="output - HRRR [K]")
    _save(fig, "2c_ii_correction.png")

    # ---- the zoom panels: Canada N of Montana ----
    lon_l, lon_r = lon[(y0 + y1) // 2, x0] - 360, lon[(y0 + y1) // 2, x1 - 1] - 360
    lat_b, lat_t = lat[y0, (x0 + x1) // 2], lat[y1 - 1, (x0 + x1) // 2]
    ext = [lon_l, lon_r, lat_b, lat_t]
    m = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
    lon_s = lon[ys, xs] - 360
    lat_s = lat[ys, xs]
    seen = m & ~heldout[ys, xs]
    hout = m & heldout[ys, xs]

    def zoom_panel(field, obs_d, cext, cb_label, title, fname):
        # background field and circles are both differences from the SAME reference in K,
        # so put them on ONE shared K scale and a single colorbar.
        fig, ax = plt.subplots(figsize=(11, 8))
        im = ax.imshow(field[y0:y1, x0:x1], cmap="RdBu_r", vmin=-cext, vmax=cext,
                       origin="lower", extent=ext, aspect="auto")
        ax.scatter(lon_s[seen], lat_s[seen], c=obs_d[ys, xs][seen], cmap="RdBu_r",
                   vmin=-cext, vmax=cext, s=140, edgecolors="k", linewidths=1.5,
                   label="METAR (seen)")
        ax.scatter(lon_s[hout], lat_s[hout], c=obs_d[ys, xs][hout], cmap="RdBu_r",
                   vmin=-cext, vmax=cext, s=170, edgecolors="lime", linewidths=2.25,
                   label="METAR (held out)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(cb_label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("lat (deg)")
        ax.legend(fontsize=9, loc="upper right")
        _save(fig, fname)

    # (iii) referenced to HRRR: the model innovation vs the obs innovation that drove it
    zoom_panel(
        resid_p, innov, zext,
        "difference from HRRR [K] — field: output−HRRR, circles: obs−HRRR",
        "2c(iii)  zoom: Canada N of Montana   |   background = model innovation, "
        "circles = METAR HRRR difference",
        "2c_iii_zoom.png")

    # (iiii) the same panel referenced to RTMA (the training target) instead of HRRR:
    # background = the model's departure from RTMA, circles = the obs' departure from RTMA.
    # Where a circle and its surroundings share a color the model followed the ob away from
    # RTMA; a colored circle on a white field means it stayed with RTMA and ignored the ob.
    zoom_panel(
        dmod, dobs, zext,
        "difference from RTMA [K] — field: output−RTMA, circles: obs−RTMA",
        "2c(iiii)  zoom: Canada N of Montana   |   background = model − RTMA, "
        "circles = obs − RTMA",
        "2c_iiii_zoom_rtma.png")

    ds.close()


def _save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=["1a", "1b", "1c", "2c"])
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # 1e/1ee/1f use edge-ness; their 1eR/1eeR/1fR twins use Laplacian roughness R.
    weight_figs = {"1a": fig_1a, "1b": fig_1b, "1c": fig_1c, "1d": fig_1d, "1dd": fig_1dd,
                   "1e": fig_1e, "1ee": fig_1ee, "1f": fig_1f,
                   "1eR": fig_1eR, "1eeR": fig_1eeR, "1fR": fig_1fR}
    if set(weight_figs) & set(args.figs):
        sd, epoch, iters = load_state()
        print(f"loaded {CKPT}  (epoch {epoch}, iters {iters})")
    for f in args.figs:
        print(f"[{f}]")
        if f in weight_figs:
            weight_figs[f](sd)
        elif f == "2c":
            fig_2c()
        else:
            raise SystemExit(f"unknown fig id {f!r}; "
                             f"valid: {sorted(weight_figs)} + ['2c']")
    print("done.")


if __name__ == "__main__":
    main()
