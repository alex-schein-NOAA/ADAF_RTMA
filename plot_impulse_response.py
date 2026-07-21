#!/usr/bin/env python
"""3a -- single-ob point-spread function (impulse response) for lowres_r4_metar_l0p3 e155.

The whole r4 architecture exists to SPREAD an observation ~30-60 cells instead of stamping a
top-hat on its own cell (memory model-is-obs-copier-not-analyzer). This measures that directly:
take the real (METAR-only) input, plant ONE extra synthetic METAR temperature ob in the most
data-sparse spot, and map how the model's output correction changes. That difference field is
the model's point-spread function -- the analysis length scale, measured, not asserted.

Needs a GPU forward pass -> run via impulse_response.sbatch.

  python plot_impulse_response.py                       # 2 isolated sites, dT = 1/3/6 K
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.YParams import YParams
import inference_parallel as ip

CONFIG = "config/params_lowres_r4_metar_l0p3.yaml"
CKPT = "ckpt_snapshots/lowres_r4_metar_l0p3_e155_17346342.tar"
SAMPLE = "/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/data_blosc_combined/test_data/2023-01-01_18.nc"
OUTDIR = "Plots/jan2023_r4_l0p3_e155/architecture"

T_RANGE = 90.0          # sta_t / hrrr_t min-max span (physical K per 2 normalized units)
KM_PER_CELL = 2.5       # RTMA grid spacing
T_CH_IN = slice(7, 10)  # sta_t input channels (3 obs-time windows)
HRRR_T_CH = 1           # hrrr_t input channel (normalized)
T_CH_OUT = 1            # output order is q, t, u10, v10
DELTAS_K = [1.0, 3.0, 6.0]
ZOOM = 90               # half-window (cells) for the PSF maps


def load_model(device):
    params = YParams(CONFIG, "EncDec")
    params.hold_out_obs = False          # keep every METAR in the baseline
    from models.encdec import build_model
    model = build_model(params).to(device).eval()
    ip.load_checkpoint(model, CKPT, device)
    return model, params


@torch.no_grad()
def forward(model, inp, device):
    x = torch.from_numpy(inp)[None].to(device).float()
    return model(x)[0].cpu().numpy()      # (4, H, W), normalized residual


def pick_station_sites(inp, n=2, sep=400):
    """Existing METAR temperature stations that are locally isolated + spaced apart.

    We perturb a REAL station (one the model already expects in its input), not a synthetic
    ob in a void, so the probe stays in-distribution -- the measured spread is the true
    operational sensitivity of the analysis to one station. Picking locally-isolated stations
    (few METAR neighbours) keeps each point-spread function clean, without OOD edge/ocean
    artifacts.
    """
    from scipy.spatial import cKDTree
    metar = inp[9] != 0                                   # analysis-time METAR temp obs
    H, W = metar.shape
    ys, xs = np.where(metar)
    pts = np.column_stack([ys, xs])
    tree = cKDTree(pts)
    nn, _ = tree.query(pts, k=2)                          # distance to nearest OTHER station
    nn = nn[:, 1]
    interior = (ys > 300) & (ys < 1050) & (xs > 450) & (xs < 1850)
    order = np.argsort(np.where(interior, nn, -1))[::-1]  # most locally-isolated interior first
    sites, chosen = [], []
    for idx in order:
        cy, cx = int(ys[idx]), int(xs[idx])
        if all((cy - qy) ** 2 + (cx - qx) ** 2 > sep ** 2 for qy, qx in chosen):
            sites.append((cy, cx)); chosen.append((cy, cx))
        if len(sites) == n:
            break
    return sites


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)
    model, params = load_model(device)
    inp, _, _ = ip.build_model_input(SAMPLE, params, include_metar=True)
    print(f"input {inp.shape}, {int((inp[9]!=0).sum())} METAR temp obs, device={device}")

    base = forward(model, inp, device)                    # (4, H, W)
    sites = pick_station_sites(inp, n=2)
    print("perturbed-station sites (row,col):", sites)

    nrow, ncol = len(sites), len(DELTAS_K) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False)
    profiles = {}

    for si, (cy, cx) in enumerate(sites):
        for di, dK in enumerate(DELTAS_K):
            pert = inp.copy()
            # nudge this EXISTING METAR's reported temperature by +dK, all 3 obs windows
            pert[T_CH_IN, cy, cx] += 2.0 * dK / T_RANGE
            out = forward(model, pert, device)
            psf = (out[T_CH_OUT] - base[T_CH_OUT]) * (T_RANGE / 2.0)   # -> Kelvin

            y0, y1 = cy - ZOOM, cy + ZOOM
            x0, x1 = cx - ZOOM, cx + ZOOM
            tile = psf[y0:y1, x0:x1]
            ax = axes[si][di]
            ext = max(abs(dK) * 1.05, np.nanpercentile(np.abs(tile), 99.9))
            im = ax.imshow(tile, cmap="RdBu_r", vmin=-ext, vmax=ext, origin="upper",
                           extent=[-ZOOM * KM_PER_CELL, ZOOM * KM_PER_CELL,
                                   -ZOOM * KM_PER_CELL, ZOOM * KM_PER_CELL])
            ax.plot(0, 0, "kx", ms=7, mew=1.5)
            ax.set_title(f"site {si+1}: +{dK:g} K ob\npeak {tile[ZOOM,ZOOM]:+.2f} K "
                         f"({tile[ZOOM,ZOOM]/dK:.2f} K/K)", fontsize=9)
            if di == 0:
                ax.set_ylabel("km")
            ax.set_xlabel("km")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            # radial profile (signed mean vs distance), normalized by dK for the +3 K case set
            yy, xx = np.indices(psf.shape)
            rr = np.hypot(yy - cy, xx - cx)
            rmax = ZOOM
            edges = np.arange(0, rmax, 3)
            prof = np.array([psf[(rr >= a) & (rr < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
            profiles[(si, dK)] = (edges[:-1] * KM_PER_CELL, prof)

        # last column: overlaid radial profiles for this site (per-K, to test linearity)
        ax = axes[si][-1]
        for dK in DELTAS_K:
            r, prof = profiles[(si, dK)]
            ax.plot(r, prof / dK, marker=".", ms=3, label=f"+{dK:g} K")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(48 * KM_PER_CELL, color="tab:green", ls="--", lw=1,
                   label="window reach (48 cells)")
        ax.set_title(f"site {si+1}: radial profile\n(response per K of ob)", fontsize=9)
        ax.set_xlabel("distance from ob (km)")
        ax.set_ylabel("mean output dT / ob dT")
        ax.legend(fontsize=7)

    fig.suptitle("3a  point-spread function: nudge ONE existing METAR's temperature by +dT and map how the "
                 "analysis correction propagates\nin-distribution sensitivity; overlapping per-K profiles "
                 "=> linear spreading, separated => the model damps larger nudges", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTDIR, "3a_impulse_response.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
