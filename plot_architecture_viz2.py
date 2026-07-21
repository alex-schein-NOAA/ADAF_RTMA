#!/usr/bin/env python
"""Architecture-explainer figures, PART 2 -- the steps plot_architecture_viz.py didn't cover.

plot_architecture_viz.py did conv_first attribution (1a), the body gate (1b), the Swin
prior (1c), and innovation-vs-correction on a real cycle (2c). This adds the mechanical
pieces of the hourglass -- the parts you can read straight off the weights or draw as a
diagram -- so the whole forward path is illustrated end to end:

  4   exact weights of every resampling conv (conv_first, the two stride-2 down convs)
  5   the stride-2 downsample MECHANISM, scaled down to a legible toy grid
  5u  5's mirror image -- the up path's PixelShuffle mechanism, same toy-grid register
  5uc 5u with the prose stripped out -- the slide plate
  6   obs-dropout: the U(0.05,0.30) rate, the rectangular blackout, real METAR realizations
  7   PixelShuffle: the channel->space rearrangement + the learned sub-pixel kernels
  8   the reconstruction tail -- conv_after_body, the head convs, conv_last per output var,
      and the down/up LayerNorm scale+bias
  9   the whole data-flow, one schematic with tensor shapes at every stage

All CPU-only. Figs 4/5/7/8/9 read only the checkpoint; fig 6 also reads one inference
netcdf for real METAR station positions (~2 min to open, that's the file, not the plot).

Run:
  python plot_architecture_viz2.py                 # everything
  python plot_architecture_viz2.py --figs 5 7 9    # a subset
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

CKPT = "ckpt_snapshots/lowres_r4_metar_l0p3_e155_17346342.tar"
INFER_NC = "inference_out/jan2023_r4_l0p3_e155/2023-01-01_18.nc"
OUTDIR = "Plots/jan2023_r4_l0p3_e155/architecture"

KM_PER_CELL = 2.5  # RTMA grid spacing

# input-channel layout (in_chans = 17), from inference_parallel.build_model_input
CH_LABELS = (
    ["hrrr_q", "hrrr_t", "hrrr_u10", "hrrr_v10"]
    + [f"sta_q[{w}]" for w in range(3)]
    + [f"sta_t[{w}]" for w in range(3)]
    + [f"sta_u10[{w}]" for w in range(3)]
    + [f"sta_v10[{w}]" for w in range(3)]
    + ["topo"]
)
OUT_VARS = ["q", "t", "u10", "v10"]  # conv_last output-channel order


def load_state():
    ck = torch.load(CKPT, map_location="cpu")
    sd = {}
    for k, v in ck["model_state"].items():
        for pre in ("_orig_mod.", "module."):
            k = k.replace(pre, "")
        sd[k] = v
    return sd, ck.get("epoch"), ck.get("iters")


def _save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ============================================================ helpers

def kernel_montage(ax, W, n_show, title, vext=None):
    """Render the n_show highest-L2-norm 3x3 kernels of a conv weight (Cout,Cin,3,3)
    packed into a tight square grid, on one shared symmetric color scale."""
    Wf = W.reshape(-1, W.shape[-2], W.shape[-1])           # (Cout*Cin, 3, 3)
    norms = np.linalg.norm(Wf.reshape(len(Wf), -1), axis=1)
    idx = np.argsort(norms)[::-1][:n_show]
    if vext is None:
        vext = np.percentile(np.abs(Wf[idx]), 99)
    ncol = int(np.ceil(np.sqrt(n_show)))
    nrow = int(np.ceil(n_show / ncol))
    k = W.shape[-1]
    canvas = np.full((nrow * (k + 1) - 1, ncol * (k + 1) - 1), np.nan)
    for j, i in enumerate(idx):
        r, c = divmod(j, ncol)
        canvas[r * (k + 1):r * (k + 1) + k, c * (k + 1):c * (k + 1) + k] = Wf[i]
    im = ax.imshow(canvas, cmap="RdBu_r", vmin=-vext, vmax=vext, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    return im, vext


# edge-ness palette, shared by 4d (and matched in plot_architecture_viz.py's 1f):
# purple = level/smoother (viridis low), amber = gradient/edge (viridis high).
SMOOTH_C = "#4c2a85"
EDGE_C = "#e0a800"
EDGE_TXT = "#9a7500"


LAPL_MAX = 6.0          # largest eigenvalue of the 4-neighbour 3x3 grid-graph Laplacian


def _edgeness(W):
    """Per-kernel edge-ness 1-|Σw|/Σ|w| for a conv weight (Cout,Cin,3,3), flattened.

    Measures DC gain: 0 = all weights share a sign (a level/smoothing filter), 1 = they
    sum to zero (a derivative filter). Ad-hoc to this repo, and permutation-invariant --
    it cannot see WHERE in the 3x3 the weights sit. See plot_architecture_viz.py's
    1d/1e header comment for the full comparison against roughness.
    """
    l1 = np.abs(W).sum(axis=(2, 3))
    return (1.0 - np.abs(W.sum(axis=(2, 3))) / (l1 + 1e-9)).ravel()


def _roughness(W, normalize=True):
    """Per-kernel Laplacian roughness R = Σ(Δw)²/Σw² over 4-neighbour adjacent pairs, for
    a conv weight (Cout,Cin,3,3), flattened.

    The Rayleigh quotient wᵀLw/wᵀw of the 3x3 grid-graph Laplacian (= normalized Dirichlet
    energy). Unlike edge-ness this IS spatially aware: exactly 0 for a constant kernel,
    LAPL_MAX for the most oscillatory one. normalize=True returns R/6, in [0,1].
    """
    dh = np.diff(W, axis=-1)
    dv = np.diff(W, axis=-2)
    num = (dh ** 2).sum(axis=(-2, -1)) + (dv ** 2).sum(axis=(-2, -1))
    r = num / ((W ** 2).sum(axis=(-2, -1)) + 1e-12)
    return (r / LAPL_MAX if normalize else r).ravel()


# canonical 3x3 kernels, as reference marks on the roughness axis (R/6)
REF_KERNELS = [(0.000, "box"), (0.111, "gauss"), (0.222, "sobel"),
               (0.667, "delta"), (0.900, "lapl")]


def edgeness_hist(ax, W, subtitle, bins=44):
    """A clean, big-text edge-ness histogram: bars colored by side of 0.5, the
    threshold line, and a smoother/edge percentage callout on each side."""
    e = _edgeness(W)
    counts, edges = np.histogram(e, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = np.where(centers > 0.5, EDGE_C, SMOOTH_C)
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=colors,
           edgecolor="white", lw=0.3, zorder=3)
    ax.axvline(0.5, color="k", ls="--", lw=1.8, zorder=4)
    frac_edge = float((e > 0.5).mean())
    ax.set_title(subtitle, fontsize=17, pad=12)
    ax.set_xlabel("edge-ness   1 − |Σw| / Σ|w|", fontsize=15)
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
    return frac_edge, len(e)


def roughness_hist(ax, r, subtitle, bins=44):
    """4d's twin panel, on Laplacian roughness R/6.

    Deliberately NOT a two-sided threshold plot like edgeness_hist: a >0.5 cut is
    arbitrary here (only 12-16% of these kernels clear it), so the bars carry a viridis
    ramp and canonical kernels are marked on the axis instead.
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
    for x, lab in REF_KERNELS:
        ax.plot([x, x], [-0.055 * top, -0.015 * top], color="0.35", lw=1.4,
                clip_on=False, zorder=6)
        ax.text(x, -0.075 * top, lab, ha="center", va="top", fontsize=9, color="0.35")

    ax.set_title(subtitle, fontsize=17, pad=12)
    ax.set_xlabel("Laplacian roughness   R = Σ(Δw)² / Σw²      (÷6)", fontsize=15,
                  labelpad=26)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color="0.85", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return mean, med


def fig_4d(sd):
    """4d -- just the two stride-2 downsample edge-ness histograms, side by side."""
    W0 = sd["down.0.weight"].float().numpy()               # (128, 96, 3, 3)
    W2 = sd["down.2.weight"].float().numpy()               # (192, 128, 3, 3)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=False)
    edgeness_hist(axL, W0, f"down.0   Conv2d(96 → 128, stride 2)\n"
                  f"full → 1/2 res · {W0.shape[0] * W0.shape[1]:,} kernels")
    edgeness_hist(axR, W2, f"down.2   Conv2d(128 → 192, stride 2)\n"
                  f"1/2 → 1/4 res · {W2.shape[0] * W2.shape[1]:,} kernels")
    axL.set_ylabel("in-out kernel pairs", fontsize=15)

    fig.suptitle("4d  edge-ness of the two stride-2 downsample convs — gradient/edge "
                 "detection (amber) vs level/smoothing (purple)", fontsize=18, y=1.02)
    _save(fig, "4d_down_edgeness.png")


def fig_4dR(sd):
    """4dR -- 4d's twin: the two stride-2 downsamples on Laplacian roughness R.

    Worth reading against 4d. The 84%/55% edge-ness gap that is 4d's headline does NOT
    reappear here (the two are ~indistinguishable) -- i.e. down.0's kernels are far more
    zero-sum than down.2's, but no spatially rougher. Zero-sum != high-frequency.
    """
    W0 = sd["down.0.weight"].float().numpy()               # (128, 96, 3, 3)
    W2 = sd["down.2.weight"].float().numpy()               # (192, 128, 3, 3)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=False)
    m0, _ = roughness_hist(axL, _roughness(W0),
                           f"down.0   Conv2d(96 → 128, stride 2)\n"
                           f"full → 1/2 res · {W0.shape[0] * W0.shape[1]:,} kernels")
    m2, _ = roughness_hist(axR, _roughness(W2),
                           f"down.2   Conv2d(128 → 192, stride 2)\n"
                           f"1/2 → 1/4 res · {W2.shape[0] * W2.shape[1]:,} kernels")
    axL.set_ylabel("in-out kernel pairs", fontsize=15)

    fig.suptitle("4dR  Laplacian roughness of the two stride-2 downsample convs — 4d's "
                 f"twin.  mean R/6 {m0:.2f} vs {m2:.2f}: near-identical, so 4d's "
                 "84%/55% split is about DC gain, not spatial frequency",
                 fontsize=15, y=1.02)
    _save(fig, "4dR_down_roughness.png")


def annotate_kernel(ax, K, title, vext=None):
    """A single 3x3 kernel as a heatmap with the exact numbers written in each cell."""
    if vext is None:
        vext = np.abs(K).max()
    ax.imshow(K, cmap="RdBu_r", vmin=-vext, vmax=vext, interpolation="nearest")
    for (r, c), v in np.ndenumerate(K):
        ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color="k" if abs(v) < 0.6 * vext else "w")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# ============================================================ 4  down-conv weights

def _conv_weight_fig(sd, key, in_labels, header, name, metric="dc"):
    W = sd[key].float().numpy()                            # (Cout, Cin, 3, 3)
    cout, cin = W.shape[:2]
    dc = W.sum(axis=(2, 3))                                # (Cout, Cin) DC gain
    norm = np.linalg.norm(W.reshape(cout, cin, -1), axis=2)

    # panel (C)'s kernel-character metric. "dc" = edge-ness (does the kernel pass DC?);
    # "rough" = Laplacian roughness R/6 (is it spatially wiggly?). Only panel C differs
    # between a fig and its R twin -- see the two _edgeness/_roughness docstrings.
    score = (_edgeness(W) if metric == "dc" else _roughness(W)).reshape(cout, cin)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.0, 1.0, 1.0],
                          height_ratios=[1, 1], wspace=0.28, hspace=0.35)

    # (A) montage of the 64 strongest 3x3 kernels -- the actual learned weights
    axm = fig.add_subplot(gs[:, 0])
    im, vext = kernel_montage(axm, W, min(64, cout * cin),
                              f"64 strongest 3x3 kernels (of {cout*cin})")
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.02, label="weight value")

    # (B) mean |kernel| -- the typical spatial footprint of this conv
    axf = fig.add_subplot(gs[0, 1])
    mk = np.abs(W).mean(axis=(0, 1))
    imf = axf.imshow(mk, cmap="viridis", interpolation="nearest")
    for (r, c), v in np.ndenumerate(mk):
        axf.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=9,
                 color="w" if v < mk.max() * 0.6 else "k")
    axf.set_xticks([]); axf.set_yticks([])
    axf.set_title("mean |weight| per tap\n(center-heavy = smoother)", fontsize=10)
    fig.colorbar(imf, ax=axf, fraction=0.046, pad=0.02)

    # (C) kernel-character split -- the one panel that differs between a fig and its twin
    axe = fig.add_subplot(gs[1, 1])
    axe.hist(score.ravel(), bins=40, range=(0, 1), color="tab:purple", alpha=0.85)
    if metric == "dc":
        axe.axvline(0.5, color="k", ls="--", lw=1)
        axe.set_title("edge-ness  1-|Σw|/Σ|w|\n"
                      f"{(score > 0.5).mean()*100:.0f}% are edge/gradient (>0.5)",
                      fontsize=10)
        axe.set_xlabel("0 = pure smoother   1 = pure edge")
    else:
        axe.axvline(np.median(score), color="k", ls="--", lw=1)
        axe.set_title("Laplacian roughness  R = Σ(Δw)²/Σw²  (÷6)\n"
                      f"median R/6 = {np.median(score):.2f}, mean {score.mean():.2f}",
                      fontsize=10)
        axe.set_xlabel("0 = constant kernel   1 = maximally oscillatory")
    axe.set_ylabel("in-out kernel pairs")

    # (D) which input channels this conv leans on (mean |DC gain| per input channel)
    axc = fig.add_subplot(gs[0, 2])
    per_in = np.abs(dc).mean(axis=0)
    order = np.argsort(per_in)[::-1]
    y = np.arange(len(order))
    axc.barh(y, per_in[order], color="tab:cyan")
    if len(order) <= 24:                                  # only label when legible
        axc.set_yticks(y)
        axc.set_yticklabels([in_labels[i] for i in order], fontsize=8)
    else:
        axc.set_yticks([])
    axc.invert_yaxis()
    axc.set_title(f"mean |DC gain| per input channel\n(sorted; {len(order)} inputs, "
                  "who this conv reads)", fontsize=10)

    # (E) kernel-norm distribution
    axn = fig.add_subplot(gs[1, 2])
    axn.hist(norm.ravel(), bins=40, color="tab:orange", alpha=0.85)
    axn.axvline(np.median(norm), color="k", ls="--", lw=1,
                label=f"median {np.median(norm):.3f}")
    axn.set_title("per-kernel L2 norm", fontsize=10)
    axn.set_xlabel("‖3x3 kernel‖")
    axn.set_ylabel("pairs")
    axn.legend(fontsize=8)

    fig.suptitle(header, fontsize=14, y=0.99)
    _save(fig, name)


_CONV_FIGS = [
    ("conv_first.weight", CH_LABELS,
     "conv_first  Conv2d(17→96, 3x3, stride 1)   |   the obs encoder, full res "
     "(exact weights)", "conv_first_kernels", "a"),
    ("down.0.weight", [f"c{c}" for c in range(96)],
     "down.0  Conv2d(96→128, 3x3, STRIDE 2)   |   full res → 1/2 res "
     "(exact weights)", "down0_kernels", "b"),
    ("down.2.weight", [f"c{c}" for c in range(128)],
     "down.2  Conv2d(128→192, 3x3, STRIDE 2)   |   1/2 → 1/4 res "
     "(exact weights)", "down2_kernels", "c"),
]


def fig_4(sd):
    for key, labels, header, stem, sub in _CONV_FIGS:
        _conv_weight_fig(sd, key, labels, f"4{sub}  {header}", f"4{sub}_{stem}.png",
                         metric="dc")


def fig_4R(sd):
    """4aR/4bR/4cR -- 4a/4b/4c's twins; identical except panel (C) is Laplacian
    roughness R instead of edge-ness."""
    for key, labels, header, stem, sub in _CONV_FIGS:
        _conv_weight_fig(sd, key, labels,
                         f"4{sub}R  {header}   [panel C: Laplacian roughness]",
                         f"4{sub}R_{stem}_roughness.png", metric="rough")


# ============================================================ 5  stride-2 mechanism

def _draw_grid(ax, n, centers=None, patch_center=None, title="", cell=1.0,
               dot_color="tab:green", face="0.96"):
    """Draw an n x n cell grid; optionally green dots at stride-2 sample centers and a
    highlighted 3x3 receptive patch."""
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((j, n - 1 - i), cell, cell, facecolor=face,
                                   edgecolor="0.75", lw=0.6))
    if patch_center is not None:
        pr, pc = patch_center
        ax.add_patch(Rectangle((pc - 1, n - 1 - (pr + 1)), 3, 3, fill=False,
                               edgecolor="tab:orange", lw=3, zorder=5))
    if centers is not None:
        for (r, c) in centers:
            ax.add_patch(plt.Circle((c + 0.5, n - 1 - r + 0.5), 0.22,
                                    color=dot_color, zorder=6))
    ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-0.3, n + 0.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def fig_5():
    """5  stride-2 receptive field, traced backwards: ONE 1/4-res cell (right) needs the
    3x3 half-res block that made it (middle), which needs the 7x7 full-res block (left).
    Almost text-free -- purple channel-depth stacks (96/128/192) + the down.0/down.2 arrows.
    (3x3 stride-2 conv: one output cell reads a 3x3 patch; a 3x3 block of half-res cells
    reads a 7x7 full-res span -> the 1 <- 3x3 <- 7x7 cascade.)"""
    CELL = 1.0
    GRID_FC, GRID_EC = "#e9eef5", "0.55"
    PURPLE = "#6a3d9a"
    YELLOW = "#ffe98a"

    def grid(ax, x0, n, yellow=None):
        y0 = -n * CELL / 2
        for i in range(n):
            for j in range(n):
                fc = YELLOW if (yellow == "all" or
                                (isinstance(yellow, set) and (i, j) in yellow)) else GRID_FC
                ax.add_patch(Rectangle((x0 + j * CELL, y0 + i * CELL), CELL, CELL,
                                       facecolor=fc, edgecolor=GRID_EC, lw=0.9))
        return x0 + n * CELL

    def stack(ax, x0, nsq, label, sq=0.5):
        y0 = -nsq * sq / 2
        for i in range(nsq):
            ax.add_patch(Rectangle((x0, y0 + i * sq), sq, sq, facecolor=PURPLE,
                                   edgecolor="white", lw=1.1))
        ax.text(x0 + sq / 2, y0 - 0.3, label, ha="center", va="top",
                fontsize=18, color=PURPLE, weight="bold")
        return x0 + sq

    def arrow(ax, x0, x1, label):
        ax.add_patch(FancyArrowPatch((x0, 0), (x1, 0), arrowstyle="-|>",
                     mutation_scale=26, color="0.25", lw=2.6))
        ax.text((x0 + x1) / 2, 0.55, label, ha="center", va="bottom",
                fontsize=17, weight="bold", color="0.2")

    fig, ax = plt.subplots(figsize=(15.5, 6.6))

    centers = {(i, j) for i in (1, 3, 5) for j in (1, 3, 5)}  # stride-2 sample centers
    x = stack(ax, 0.0, 6, "96")                 # full-res f0 depth
    x = grid(ax, x + 0.8, 7, yellow=centers)     # 7x7 full-res block, strides highlighted
    arrow(ax, x + 0.6, x + 2.4, "down.0")
    x = stack(ax, x + 3.0, 8, "128")            # 1/2-res depth
    x = grid(ax, x + 0.8, 3, yellow="all")       # 3x3 half-res block (all are stride outputs)
    arrow(ax, x + 0.6, x + 2.4, "down.2")
    x = stack(ax, x + 3.0, 12, "192")           # 1/4-res depth
    x = grid(ax, x + 0.8, 1)                     # the single 1/4-res cell

    ax.set_xlim(-0.3, x + 0.3)
    ax.set_ylim(-4.1, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "5_stride2_mechanism.png", tight=False)


# ============================================================ 5u  pixelshuffle mechanism
#
# Fig 5's mirror image: 5 traces the stride-2 stem backwards (1 <- 3x3 <- 7x7), 5u traces
# the up path forwards (1 -> 2x2 -> 4x4), in the same near-text-free register. Fig 7 covers
# the same operator analytically (real learned sub-pixel kernels, shape table); this is the
# slide version, and it exists to answer one question -- where the fine grid comes from.
#
# The answer the drawing makes visible: PixelShuffle has NO parameters. It is a permutation,
# so it cannot add information. The four fine pixels are manufactured by the conv BEFORE it,
# at coarse resolution, in the channel dimension.
#
# NOTE this is deliberately NOT called an "upscale": params.upscale is 1 and LowResEncDec
# raises if it isn't (models/encdec.py:1265). The feature map is upsampled 4x; the field is
# not upscaled -- in 1356x2294, out 1356x2294.

# PixelShuffle(2) is out[c, 2h+i, 2w+j] = in[4c + 2i + j, h, w], so group k = 2i+j lands at
# sub-position (i, j): 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right. Same order as
# fig 7A's `pos`. VERIFIED against torch, not read off the docs:
#   nn.PixelShuffle(2)(arange(8).reshape(1,8,1,1))  ->  ch0 = [[0,1],[2,3]], ch1 = [[4,5],[6,7]]
#
# The grouping is INTERLEAVED at period 4, not blocked. Consecutive channels 4c..4c+3 are the
# four sub-pixels of ONE output channel c; it is NOT "the first 96 channels become sub-pixel
# 0". An earlier draft of this figure drew four contiguous 96-channel blocks and was wrong.
# That also means the group structure is invisible at any coarse-graining of the channel axis
# (a square worth 16 channels already spans four full cycles), which is why the interleave is
# shown in a magnified detail box rather than by colouring the main stacks.
#
# Palette = dataviz categorical slots 1/2/3/7 (blue, orange, aqua, violet). All four groups
# are on screen at once, so they were validated against the ALL-PAIRS list on a white
# surface, not the adjacent list: worst CVD dE 9.2 (target 8), worst normal-vision dE 16.3
# (floor 15). Slots 1/2/3/5 and 1/2/3/4 both FAIL normal-vision all-pairs (12.9, 13.7) --
# don't "improve" the colours without re-running the validator. Aqua sits at 2.8:1 contrast,
# under the 3:1 bar, so the relief rule applies: every group is also labelled with its index
# on the stack AND on the grid, and identity is never carried by colour alone.
SUBPIX_C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
CH_PER_SQ = 16          # one stack square = 16 channels, the same scale fig 5 uses


def fig_5u(clean=False):
    """clean=True strips 5u down to a slide plate: the only text left is the channel
    counts, the resolution under each grid, the two conv names, PixelShuffle 2, the
    magnified interleave's 0-3 with a small grey 4c/+1/+2/+3 under it, and the words
    "the interleave, magnified". Everything the annotated version explains in prose
    is gone -- the caller says it out loud instead."""
    CELL = 1.0
    GRID_FC, GRID_EC = "#e9eef5", "0.55"
    PURPLE = "#6a3d9a"

    def grid(ax, x0, n, fills=None, blocks=None):
        """n x n grid. fills maps (row_from_top, col) -> colour; blocks outlines each 2x2
        block in its stage-1 group colour."""
        y0 = -n * CELL / 2
        for r in range(n):                       # r counts DOWN from the top
            for c in range(n):
                y = y0 + (n - 1 - r) * CELL
                fc = GRID_FC if fills is None else fills.get((r, c), GRID_FC)
                ax.add_patch(Rectangle((x0 + c * CELL, y), CELL, CELL, facecolor=fc,
                                       edgecolor=GRID_EC, lw=0.9, zorder=2))
                if fills is not None:
                    ax.text(x0 + (c + 0.5) * CELL, y + CELL / 2,
                            str(2 * (r % 2) + (c % 2)), ha="center", va="center",
                            fontsize=13, color="white", weight="bold", zorder=4)
        if blocks:
            for (br, bc), col in blocks.items():
                y = y0 + (n - 2 - 2 * br) * CELL
                # white halo first -- a group-coloured outline on a saturated group-coloured
                # fill is invisible without it
                for ec, lw, z in ((("white"), 6.0, 5), (col, 3.2, 6)):
                    ax.add_patch(Rectangle((x0 + 2 * bc * CELL, y), 2 * CELL, 2 * CELL,
                                           fill=False, edgecolor=ec, lw=lw, zorder=z))
        return x0 + n * CELL

    def stack(ax, x0, nch, label, sq=0.25):
        """Vertical channel-depth stack, one square per CH_PER_SQ channels.

        Deliberately NOT coloured by sub-pixel group: the groups interleave with period 4,
        so one square (16 channels) spans four whole cycles and no colouring of it could be
        honest. The interleave lives in `detail()` instead.
        """
        nsq = nch // CH_PER_SQ
        y0 = -nsq * sq / 2
        for i in range(nsq):
            ax.add_patch(Rectangle((x0, y0 + i * sq), sq, sq, facecolor=PURPLE,
                                   edgecolor="white", lw=0.9, zorder=3))
        ax.text(x0 + sq / 2, y0 - 0.32, label, ha="center", va="top",
                fontsize=16, color=PURPLE, weight="bold")
        return x0 + sq

    def detail(ax, xc, yc):
        """Magnified callout: the true period-4 interleave, for one output channel c."""
        s, gap = 0.95, 0.16
        w4 = 4 * s + 3 * gap
        total = w4 + 1.9 + 2 * s
        x0 = xc - total / 2
        for k in range(4):                       # 4 CONSECUTIVE input channels
            xk = x0 + k * (s + gap)
            ax.add_patch(Rectangle((xk, yc - s / 2), s, s, facecolor=SUBPIX_C[k],
                                   edgecolor="white", lw=1.2, zorder=3))
            ax.text(xk + s / 2, yc, str(k), ha="center", va="center", fontsize=15,
                    weight="bold", color="white", zorder=4)
            # short forms: "4c+1" at this size overruns the 0.95-wide square and collides
            ax.text(xk + s / 2, yc - s / 2 - 0.16, "4c" if k == 0 else f"+{k}",
                    ha="center", va="top", fontsize=9.5 if clean else 11, color="0.45")
        xa = x0 + w4 + 0.3
        ax.add_patch(FancyArrowPatch((xa, yc), (xa + 1.3, yc), arrowstyle="-|>",
                     mutation_scale=20, color="0.25", lw=2.2))
        xb = x0 + w4 + 1.9
        for k in range(4):                       # -> the 2x2 of ONE output channel
            r, c = divmod(k, 2)
            ax.add_patch(Rectangle((xb + c * s, yc + s / 2 - (r + 1) * s), s, s,
                                   facecolor=SUBPIX_C[k], edgecolor="white", lw=1.2,
                                   zorder=3))
            ax.text(xb + (c + 0.5) * s, yc + s / 2 - (r + 0.5) * s, str(k), ha="center",
                    va="center", fontsize=15, weight="bold", color="white", zorder=4)
        if clean:
            ax.text(xc, yc + s + 0.34, "the interleave, magnified", ha="center",
                    va="bottom", fontsize=12, color="0.45")
            return
        ax.text(xc, yc + s + 0.45, "the interleave, magnified — 4 CONSECUTIVE channels "
                "make ONE output channel's 2x2", ha="center", va="bottom", fontsize=13.5,
                weight="bold", color="0.2")
        ax.text(xc, yc - s - 0.95, "384 = 96 such groups. It is NOT \"the first 96 "
                "channels become sub-pixel 0\".", ha="center", va="top",
                fontsize=12, style="italic", color="0.4")

    def arrow(ax, x0, x1, label, sub=None, fs=16):
        ax.add_patch(FancyArrowPatch((x0, 0), (x1, 0), arrowstyle="-|>",
                     mutation_scale=26, color="0.25", lw=2.6))
        ax.text((x0 + x1) / 2, 0.5, label, ha="center", va="bottom",
                fontsize=fs, weight="bold", color="0.2")
        if sub and not clean:
            ax.text((x0 + x1) / 2, -0.55, sub, ha="center", va="top",
                    fontsize=11, style="italic", color="0.4", linespacing=1.35)

    def res_label(ax, xc, y, txt):
        ax.text(xc, y, txt, ha="center", va="top", fontsize=13, color="0.35")

    fig, ax = plt.subplots(figsize=(17.5, 4.6 if clean else 6.4))

    quad = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}          # (row, col) -> group
    # the "= 96 x 4 / still 1/4 res" second line is annotation, not a channel count
    lbl384 = (lambda res: "384" if clean else f"384 = 96 x 4\nstill {res} res")
    # NOT narrowed for clean mode: dropping the sub-label frees VERTICAL space, but the
    # constraint here is horizontal -- "PixelShuffle 2" is ~5.8 data units wide and
    # overruns the flanking stacks at any tighter spacing.
    shuf_gap, stack_gap = 5.0, 6.6

    # ---- station A: body output, 1/4 res ----
    x = stack(ax, 0.0, 192, "192")
    xa = x + 0.8
    x = grid(ax, xa, 1)
    res_label(ax, (xa + x) / 2, -1.0, "1/4 res")

    # ---- up.0 conv: still 1/4 res, but now 96 interleaved groups of 4 ----
    arrow(ax, x + 0.8, x + 3.0, "up.0", "conv 3x3")
    x = stack(ax, x + 3.8, 384, lbl384("1/4"))

    # ---- shuffle 1 ----
    arrow(ax, x + 1.5, x + shuf_gap, "PixelShuffle 2", "0 params\npure rearrangement", fs=15)
    x = stack(ax, x + stack_gap, 96, "96")
    xb = x + 0.8
    x = grid(ax, xb, 2, fills={rc: SUBPIX_C[g] for rc, g in quad.items()})
    res_label(ax, (xb + x) / 2, -1.5, "1/2 res")

    # ---- up.3 conv ----
    arrow(ax, x + 0.8, x + 3.0, "up.3", "conv 3x3")
    x = stack(ax, x + 3.8, 384, lbl384("1/2"))

    # ---- shuffle 2: every 1/2-res cell becomes its own 2x2 ----
    arrow(ax, x + 1.5, x + shuf_gap, "PixelShuffle 2", "0 params\npure rearrangement", fs=15)
    x = stack(ax, x + stack_gap, 96, "96")
    xc = x + 0.8
    x = grid(ax, xc, 4, fills={(r, c): SUBPIX_C[2 * (r % 2) + (c % 2)]
                               for r in range(4) for c in range(4)},
             blocks={rc: SUBPIX_C[g] for rc, g in quad.items()})
    res_label(ax, (xc + x) / 2, -2.5, "full res")

    # centred on the whole cascade, not on either shuffle: it is a fact about both, and
    # under shuffle 1 (left of centre) its heading overruns the left edge
    detail(ax, x / 2, -6.2 if clean else -7.3)

    if not clean:
        ax.text(x / 2, 3.9, "the fine grid comes out of the CHANNEL axis: the conv predicts "
                "4 values per output pixel at COARSE resolution,\nand PixelShuffle only "
                "moves them into the 2x2 - no information is created", ha="center",
                va="bottom", fontsize=13.5, color="0.25")
        ax.text(x, -10.15, f"one square = {CH_PER_SQ} channels (as in fig 5)   |   digits = "
                "sub-pixel group k = 2i+j   |   block outlines = which 1/2-res cell each 2x2 "
                "came from", ha="right", va="top", fontsize=10.5, color="0.45")

    ax.set_xlim(-0.3, x + 0.3)
    ax.set_ylim(-7.5 if clean else -10.7, 3.6 if clean else 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, f"5u{'c' if clean else ''}_pixelshuffle_mechanism.png", tight=False)


def fig_5uc():
    """5uc -- 5u stripped to a slide plate (see fig_5u's clean=True)."""
    fig_5u(clean=True)


# ============================================================ 6  obs dropout

# training-dropout hyperparameters (from config/params_lowres_r4_metar_l0p3.yaml)
R_MIN, R_MAX = 0.05, 0.30
VALID_RATIO = 0.10
BLOCK_PROB = 0.30
BLOCK_MIN, BLOCK_MAX = 64, 192   # cells


def _sim_heldout(station_yx, ny, nx, rng, train=True):
    """Replicate dataloader_multifiles._heldout_mask on a set of station (y,x) cells."""
    n = len(station_yx)
    ratio = rng.uniform(R_MIN, R_MAX) if train else VALID_RATIO
    k = int(n * ratio)
    held = set(rng.choice(n, size=k, replace=False).tolist()) if k else set()
    held_thin = np.zeros(n, dtype=bool)
    for h in held:
        held_thin[h] = True
    block = None
    held_block = np.zeros(n, dtype=bool)
    if train and rng.random() < BLOCK_PROB:
        bh = int(rng.integers(BLOCK_MIN, BLOCK_MAX + 1))
        bw = int(rng.integers(BLOCK_MIN, BLOCK_MAX + 1))
        y0 = int(rng.integers(0, max(ny - bh, 0) + 1))
        x0 = int(rng.integers(0, max(nx - bw, 0) + 1))
        block = (x0, y0, bw, bh)
        ys, xs = station_yx[:, 0], station_yx[:, 1]
        held_block = (ys >= y0) & (ys < y0 + bh) & (xs >= x0) & (xs < x0 + bw)
    return held_thin, held_block, block, ratio


def fig_6():
    import xarray as xr
    ds = xr.open_dataset(INFER_NC)
    obs_source = ds["obs_source"].to_numpy()
    ny, nx = obs_source.shape
    ys, xs = np.where(obs_source == 2)                    # METAR cells (train_obs_source: metar)
    station_yx = np.column_stack([ys, xs])
    ds.close()
    n_sta = len(station_yx)

    # ---- 6a: the dropout distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    xr_ = np.linspace(0, 0.35, 400)
    band = np.where((xr_ >= R_MIN) & (xr_ <= R_MAX), 1.0 / (R_MAX - R_MIN), 0.0)
    ax.fill_between(xr_ * 100, band, color="tab:blue", alpha=0.35,
                    label=f"train: r ~ U({R_MIN:g},{R_MAX:g})")
    ax.axvline(VALID_RATIO * 100, color="tab:red", lw=2,
               label=f"valid: fixed {VALID_RATIO:.0%}")
    ax.set_xlabel("per-sample held-out fraction of stations (%)")
    ax.set_ylabel("density")
    ax.set_title("random-thinning rate\n(density-aware, not one tuned fraction)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    sides_km = np.arange(BLOCK_MIN, BLOCK_MAX + 1) * KM_PER_CELL
    ax.fill_between(sides_km, np.ones_like(sides_km) / len(sides_km),
                    step="mid", color="tab:green", alpha=0.4)
    ax.set_xlabel("blackout side length (km)")
    ax.set_ylabel("prob (per side)")
    ax.set_title(f"rectangular blackout side ~ U({BLOCK_MIN},{BLOCK_MAX}) cells\n"
                 f"= {BLOCK_MIN*KM_PER_CELL:.0f}–{BLOCK_MAX*KM_PER_CELL:.0f} km, "
                 f"applied with prob {BLOCK_PROB:.0%}", fontsize=10)

    ax = axes[2]
    rng = np.random.default_rng(0)
    fracs, had_block = [], []
    for _ in range(4000):
        ht, hb, blk, _ = _sim_heldout(station_yx, ny, nx, rng, train=True)
        fracs.append((ht | hb).mean())
        had_block.append(blk is not None)
    fracs = np.array(fracs); had_block = np.array(had_block)
    ax.hist(fracs[~had_block] * 100, bins=40, color="tab:blue", alpha=0.7,
            label="thinning only")
    ax.hist(fracs[had_block] * 100, bins=40, color="tab:green", alpha=0.7,
            label="thinning + blackout")
    ax.set_xlabel("actual % of METAR held out (per sample)")
    ax.set_ylabel("samples")
    ax.set_title(f"effective held-out fraction\n(real METAR density, n={n_sta} stations)",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("6a  obs-dropout regime: what the model is forced to reconstruct each step "
                 "(the interpolation pressure the RTMA-emulation loss otherwise starves)",
                 fontsize=13)
    _save(fig, "6a_holdout_distributions.png")

    # ---- 6b: four real realizations over the METAR network ----
    # auto-select seeds against the REAL station layout so the blackout examples land on
    # land (a random box often falls on ocean/edge and captures no METAR).
    noblk, withblk = [], []
    for s in range(500):
        rng = np.random.default_rng(s)
        ht, hb, blk, ratio = _sim_heldout(station_yx, ny, nx, rng, train=True)
        if blk is None:
            noblk.append((s, ratio))
        else:
            withblk.append((s, int(hb.sum()), ratio, blk[2] * blk[3]))
    noblk.sort(key=lambda t: t[1])
    withblk.sort(key=lambda t: -t[1])                     # most stations captured first
    big = withblk[0]                                      # biggest capture (large box)
    small = min(withblk[:40], key=lambda t: t[3])         # a smaller box that still hits land
    seeds = [noblk[0][0], noblk[-1][0], big[0], small[0]]  # light / heavy / big / small blackout

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, s in zip(axes.ravel(), seeds):
        rng = np.random.default_rng(s)
        ht, hb, blk, ratio = _sim_heldout(station_yx, ny, nx, rng, train=True)
        held = ht | hb
        ax.scatter(xs[~held], ys[~held], s=5, c="0.55", label="seen by model")
        ax.scatter(xs[ht & ~hb], ys[ht & ~hb], s=12, c="tab:orange",
                   label="held out (thinning)")
        if blk is not None:
            x0, y0, bw, bh = blk
            ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False, edgecolor="tab:red",
                                   lw=2.2))
            ax.scatter(xs[hb], ys[hb], s=16, c="tab:red", label="held out (blackout)")
        ttl = f"r={ratio:.0%} thinned"
        ttl += f"  +  {bw*KM_PER_CELL:.0f}x{bh*KM_PER_CELL:.0f} km blackout" if blk else "  (no blackout this draw)"
        ax.set_title(ttl, fontsize=11)
        ax.set_xlim(0, nx); ax.set_ylim(0, ny)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, loc="lower left", markerscale=1.5)

    fig.suptitle("6b  four training-dropout draws on the real METAR network — orange = random "
                 "thinning, red box = the long-range blackout that forces genuine interpolation",
                 fontsize=13)
    _save(fig, "6b_holdout_realizations.png")


# ============================================================ 7  pixelshuffle

def fig_7(sd):
    # PixelShuffle(2): out[c, 2h+i, 2w+j] = in[4c + 2i + j, h, w].
    # up.0: Conv2d(192 -> 384) then PixelShuffle(2) -> 96 ch at 2x res.
    W = sd["up.0.weight"].float().numpy()                 # (384, 192, 3, 3)
    num_feat = 96

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.3)

    # (A) schematic: 4 source channels -> a 2x2 output block
    axs = fig.add_subplot(gs[0, 0]); axs.axis("off")
    axs.set_xlim(0, 10); axs.set_ylim(0, 10)
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    labels = ["4c+0", "4c+1", "4c+2", "4c+3"]
    # stack of 4 source channels on the left
    for i, (col, lab) in enumerate(zip(colors, labels)):
        axs.add_patch(FancyBboxPatch((0.4, 7.2 - 1.6 * i), 2.2, 1.2,
                      boxstyle="round,pad=0.02", facecolor=col, alpha=0.85))
        axs.text(1.5, 7.8 - 1.6 * i, f"ch {lab}", ha="center", va="center",
                 color="w", fontsize=10, weight="bold")
    # target 2x2 block on the right
    pos = {0: (6.4, 6.0), 1: (8.0, 6.0), 2: (6.4, 4.4), 3: (8.0, 4.4)}
    for i, col in enumerate(colors):
        x, y = pos[i]
        axs.add_patch(Rectangle((x, y), 1.6, 1.6, facecolor=col, alpha=0.85,
                                edgecolor="k"))
        axs.add_patch(FancyArrowPatch((2.7, 7.8 - 1.6 * i), (x + 0.1, y + 0.8),
                      arrowstyle="->", mutation_scale=13, color="0.4", lw=1.4))
    axs.text(7.2, 3.9, "output channel c\n(2x2 spatial block)", ha="center", va="top",
             fontsize=10)
    axs.text(1.5, 8.9, "the 4 source channels\nfor output c", ha="center", va="bottom",
             fontsize=10)
    axs.set_title("A. PixelShuffle(2): channels → space\n"
                  "out[c, 2h+i, 2w+j] = in[4c+2i+j, h, w]", fontsize=10)

    # (B) the REAL learned sub-pixel kernels for a chosen output channel c.
    # Pick the output feature channel whose 4 source filters have the largest total norm.
    grp_norm = np.sqrt((W.reshape(num_feat, 4, 192, 3, 3) ** 2).sum(axis=(2, 3, 4))).sum(axis=1)
    c = int(np.argmax(grp_norm))
    axb = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    # collapse the 192 input channels by summed-DC to a single representative 3x3 per sub-pixel
    sub = W.reshape(num_feat, 4, 192, 3, 3)[c]            # (4, 192, 3, 3)
    subk = sub.sum(axis=1)                                # (4,3,3) DC-summed over inputs
    vext = np.abs(subk).max()
    # 2x2 layout matching pos: (i,j) = (0,0),(0,1),(1,0),(1,1)
    inner = axb[0].get_gridspec()  # not used; draw as a 2x2 imshow montage
    axb[0].remove(); axb[1].remove()
    sub_gs = gs[0, 1:3].subgridspec(2, 2, wspace=0.1, hspace=0.42)
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for k in range(4):
        rr, cc = order[k]
        a = fig.add_subplot(sub_gs[rr, cc])
        annotate_kernel(a, subk[k], f"sub-pixel ({rr},{cc})  ← ch 4c+{k}", vext)
    fig.text(0.72, 0.95, f"B. up.0's learned sub-pixel kernels for output channel c={c} "
             "(3x3 summed over 192 inputs)\nthe four filters that paint one 2x2 output block",
             ha="center", va="top", fontsize=10)

    # (C) the two-stage upsample as shapes
    axc = fig.add_subplot(gs[1, :]); axc.axis("off")
    axc.set_xlim(0, 12); axc.set_ylim(0, 4)
    stages = [
        ("body out\n192 ch\n1/4 res", 0.6, "tab:blue"),
        ("up.0 conv\n→ 384 ch\n1/4 res", 2.9, "tab:blue"),
        ("PixelShuffle2\n→ 96 ch\n1/2 res", 5.2, "tab:green"),
        ("up.3 conv\n→ 384 ch\n1/2 res", 7.5, "tab:blue"),
        ("PixelShuffle2\n→ 96 ch\nFULL res", 9.8, "tab:green"),
    ]
    for i, (txt, x, col) in enumerate(stages):
        h = [1.0, 1.0, 2.0, 2.0, 4.0][i] * 0.45 + 0.6     # box height ~ resolution
        axc.add_patch(FancyBboxPatch((x, 2 - h / 2), 1.7, h, boxstyle="round,pad=0.03",
                      facecolor=col, alpha=0.25, edgecolor=col, lw=1.5))
        axc.text(x + 0.85, 2, txt, ha="center", va="center", fontsize=9)
        if i:
            axc.add_patch(FancyArrowPatch((x - 0.55, 2), (x - 0.02, 2),
                          arrowstyle="->", mutation_scale=15, color="0.4"))
    axc.text(6, 3.7, "C. the up path: each PixelShuffle trades 4x channels for 2x2 space, "
             "1/4 res → full res (no interpolation weights — the upsampling filter is learned)",
             ha="center", fontsize=10)

    fig.suptitle("7  PixelShuffle upsampling — how the 1/4-res body output is painted back "
                 "to full resolution", fontsize=14, y=0.98)
    _save(fig, "7_pixelshuffle.png", tight=False)


# ============================================================ 8  reconstruction tail

def fig_8(sd):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.4)

    # (A) conv_after_body montage
    axa = fig.add_subplot(gs[0, 0])
    im, _ = kernel_montage(axa, sd["conv_after_body.weight"].float().numpy(), 64,
                           "conv_after_body 3x3 (192→192)\ngated by LayerScale, added at 1/4 res")
    fig.colorbar(im, ax=axa, fraction=0.046, pad=0.02)

    # (B) head.0 + head.2 montage
    axh = fig.add_subplot(gs[0, 1])
    im, _ = kernel_montage(axh, sd["head.0.weight"].float().numpy(), 64,
                           "head.0 3x3 (192→96)\nfull-res fuse of cat(up-feat, f0 skip)")
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.02)

    axh2 = fig.add_subplot(gs[0, 2])
    im, _ = kernel_montage(axh2, sd["head.2.weight"].float().numpy(), 64,
                           "head.2 3x3 (96→96)\nsecond full-res mixing conv")
    fig.colorbar(im, ax=axh2, fraction=0.046, pad=0.02)

    # (C) conv_last: the four output-variable kernels' spatial footprint + input reliance
    W = sd["conv_last.weight"].float().numpy()            # (4, 96, 3, 3)
    axl = fig.add_subplot(gs[1, 0])
    mk = np.abs(W).mean(axis=1)                           # (4,3,3) mean footprint per var
    # stack the 4 footprints side by side
    canvas = np.full((3, 4 * 4 - 1), np.nan)
    for v in range(4):
        canvas[:, v * 4:v * 4 + 3] = mk[v]
    iml = axl.imshow(canvas, cmap="viridis", interpolation="nearest")
    axl.set_xticks([v * 4 + 1 for v in range(4)])
    axl.set_xticklabels(OUT_VARS, fontsize=11, weight="bold")
    axl.set_yticks([])
    axl.set_title("conv_last (96→4): mean |weight|\nfootprint per output var", fontsize=10)
    fig.colorbar(iml, ax=axl, fraction=0.02, pad=0.02)

    # DC gain magnitude per output var (how strongly conv_last drives each variable)
    axd = fig.add_subplot(gs[1, 1])
    dc = np.abs(W.sum(axis=(2, 3)))                       # (4,96)
    axd.bar(OUT_VARS, dc.sum(axis=1), color=["tab:cyan", "tab:red", "tab:blue", "tab:green"])
    axd.set_title("conv_last total |DC gain| per output var\n(summed over 96 head channels)",
                  fontsize=10)
    axd.set_ylabel("Σ |Σ 3x3|")

    # (D) the down/up LayerNorm scale (weight) per channel, sorted -- what each resample amplifies
    axn = fig.add_subplot(gs[1, 2])
    for key, lab, col in [
        ("down.1.0.norm.weight", "down.1 (1/2, 128ch)", "tab:blue"),
        ("down.3.0.norm.weight", "down.3 (1/4, 192ch)", "tab:purple"),
        ("up.2.0.norm.weight", "up.2 (1/2, 96ch)", "tab:green"),
        ("up.5.0.norm.weight", "up.5 (full, 96ch)", "tab:orange"),
    ]:
        g = np.sort(sd[key].float().numpy())[::-1]
        axn.plot(np.linspace(0, 1, len(g)), g, label=lab, color=col, lw=1.8)
    axn.axhline(1.0, color="k", ls="--", lw=1, label="init = 1")
    axn.set_xlabel("channel rank (fraction)")
    axn.set_ylabel("LayerNorm γ (scale)")
    axn.set_title("post-resample LayerNorm scale per channel\n(sorted; the learned per-channel gain)",
                  fontsize=10)
    axn.legend(fontsize=7)

    fig.suptitle("8  the reconstruction tail — conv_after_body, the two full-res head convs, "
                 "conv_last per output variable, and the resample-path norms", fontsize=14, y=0.98)
    _save(fig, "8_reconstruction_tail.png")


# ============================================================ 9  data-flow diagram

def fig_9(sd):
    # counts from the checkpoint so the diagram can't drift from the real weights
    def np_(k):
        return sd[k].float().numpy()
    # real parameters only -- exclude the float attn_mask buffers (and int index buffers)
    n_params = sum(int(np.prod(v.shape)) for k, v in sd.items()
                   if v.dtype.is_floating_point and "attn_mask" not in k and k != "mean")

    fig, ax = plt.subplots(figsize=(17, 8.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, txt, col, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                     facecolor=col, alpha=0.30, edgecolor=col, lw=1.6))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, col="0.35", style="->", lw=1.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=15, color=col, lw=lw))

    C_STEM, C_DOWN, C_BODY, C_UP, C_HEAD = ("tab:gray", "tab:blue", "tab:red",
                                            "tab:green", "tab:orange")
    yb = 30  # main encoder row baseline

    # encoder row (full -> 1/4)
    box(2, yb, 12, 8, "input\n(1, 17, 1356, 2294)\n4 HRRR + 12 obs + topo", C_STEM)
    box(17, yb, 11, 8, "conv_first\n17→96, 3x3\nfull res  → f0", C_STEM)
    box(31, yb, 11, 8, "down.0 +LN+LReLU\n96→128, s2\n1/2 res", C_DOWN)
    box(45, yb, 11, 8, "down.2 +LN+LReLU\n128→192, s2\n1/4 res  → fc", C_DOWN)
    for x0, x1 in [(14, 17), (28, 31), (42, 45)]:
        arrow(x0, yb + 4, x1, yb + 4)

    # body (down a row on the right, 1/4 res)
    yc = 15
    box(45, yc, 11, 9,
        "BODY  2×RSTB\n(4+4 Swin blocks)\n192 ch @ 1/4 res\nwindow 12 = 48 cells\n"
        "the only attention", C_BODY, fs=8)
    arrow(50.5, yb, 50.5, yc + 9)                          # fc down into body
    box(31, yc, 12, 9,
        "conv_after_body\n192→192, 3x3\n× LayerScale γ\n  + fc  (skip)", C_BODY, fs=8)
    arrow(45, yc + 4.5, 43, yc + 4.5)                      # body -> conv_after_body
    ax.text(38, yc + 10.2, "residual, gated", fontsize=7, color=C_BODY, ha="center")

    # decoder row (1/4 -> full), going left
    box(17, yc, 12, 9, "up.0 conv→384\nPixelShuffle2\n+LN+LReLU\n→96 @ 1/2 res", C_UP, fs=8)
    box(2, yc, 12, 9, "up.3 conv→384\nPixelShuffle2\n+LN+LReLU\n→96 @ FULL res", C_UP, fs=8)
    arrow(31, yc + 4.5, 29, yc + 4.5)
    arrow(17, yc + 4.5, 14, yc + 4.5)

    # head row (full res, bottom)
    yh = 2
    box(2, yh, 15, 8, "cat( up-feat 96 , f0 96 )\n= 192 ch, full res", C_HEAD, fs=8)
    box(20, yh, 12, 8, "head.0 →96\nhead.2 →96\n3x3, LReLU", C_HEAD, fs=8)
    box(35, yh, 12, 8, "conv_last\n96→4, 3x3\nq, t, u10, v10", C_HEAD, fs=8)
    box(50, yh, 14, 8, "+ HRRR (residual)\n→ analysis\n(1, 4, 1356, 2294)", C_STEM, fs=8)
    arrow(8, yc, 8, yh + 8)                                # up.3 -> concat
    # f0 skip: run it down the clear gap between up.3 (≤14) and up.0 (≥17)
    arrow(18, yb, 13, yh + 8, col=C_STEM, style="->", lw=1.4)
    ax.text(4, 27.5, "f0 skip: 3x3 conv\n(can spread an ob)", fontsize=7,
            color=C_STEM, va="center", ha="left")
    for x0, x1 in [(17, 20), (32, 35), (47, 50)]:
        arrow(x0, yh + 4, x1, yh + 4)

    ax.text(50, 47, "9  LowResEncDec forward pass — full data flow "
            f"(r4 Tier-M, {n_params/1e6:.2f} M params)", ha="center", fontsize=15)
    ax.text(50, 44, "hourglass: convs downsample to 1/4 res → attention body does the "
            "spreading cheaply → PixelShuffle upsamples → a 3x3 head fuses the full-res skip. "
            "Residual vs HRRR.", ha="center", fontsize=9.5, color="0.3")

    _save(fig, "9_dataflow.png", tight=False)


# ============================================================ main

FIGS = {"4": fig_4, "4R": fig_4R, "4d": fig_4d, "4dR": fig_4dR, "5": fig_5, "5u": fig_5u, "5uc": fig_5uc,
        "6": fig_6, "7": fig_7, "8": fig_8, "9": fig_9}
NEEDS_WEIGHTS = {"4", "4R", "4d", "4dR", "7", "8", "9"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=list(FIGS))
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    sd = None
    if NEEDS_WEIGHTS & set(args.figs):
        sd, epoch, iters = load_state()
        print(f"loaded {CKPT}  (epoch {epoch}, iters {iters})")
    for f in args.figs:
        print(f"[{f}]")
        if f in NEEDS_WEIGHTS:
            FIGS[f](sd)
        else:
            FIGS[f]()
    print("done.")


if __name__ == "__main__":
    main()
