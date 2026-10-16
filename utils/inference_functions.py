import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import datetime as dt
import hdf5plugin

from utils.misc_functions import *
from utils.YParams import *

##################


# ============================================================================
# HELPER FUNCTIONS & CHECKPOINT LOADING
# ============================================================================

def load_checkpoint_weights(model, checkpoint_file, map_device):
    """Loads model weights, iteratively stripping 'module.' and '_orig_mod.' prefixes,
    and filtering out resolution-dependent 'attn_mask' buffers.
    """
    ck = torch.load(checkpoint_file, map_location=map_device)
    state = ck.get("model_state", ck.get("state_dict"))
    if state is None:
        raise KeyError("Checkpoint does not contain 'model_state' or 'state_dict'.")

    def _strip(k):
        changed = True
        while changed:
            changed = False
            for pre in ("_orig_mod.", "module."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        return k

    #Filter out 'attn_mask' entries alongside key stripping
    clean_state = {
        _strip(key): val 
        for key, val in state.items() 
        if "attn_mask" not in key
    }

    # Set strict=False to allow skipping filtered attention masks
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    return ck, missing, unexpected


def load_stats(stats_path, var_names):
    """Extracts min and max values for specified variables from stats.csv."""
    stats_df = pd.read_csv(stats_path).set_index("variable")
    vmin = np.array([stats_df.loc[v, "min"] for v in var_names], dtype=np.float32)
    vmax = np.array([stats_df.loc[v, "max"] for v in var_names], dtype=np.float32)
    return vmin, vmax


def reverse_norm(arr, vmin, vmax, channel_axis=0):
    """Reverses min-max normalization from [-1, 1] back to physical units."""
    arr = np.asarray(arr, dtype=np.float32)
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    if channel_axis < 0:
        channel_axis = arr.ndim + channel_axis

    if arr.shape[channel_axis] != vmin.size:
        raise ValueError(
            f"Channel mismatch: axis {channel_axis} has {arr.shape[channel_axis]} channels "
            f"but stats vector has {vmin.size}."
        )

    reshape = [1] * arr.ndim
    reshape[channel_axis] = vmin.size
    vmin_b = vmin.reshape(reshape)
    vmax_b = vmax.reshape(reshape)
    return (arr + 1.0) * (vmax_b - vmin_b) / 2.0 + vmin_b


# ============================================================================
# DATA PREPARATION & MODEL INFERENCE
# ============================================================================

def build_model_input_from_netcdf(nc_file, p, include_metar=True, deny_goes_channels=None, deny_goes_hours=None,
                                  deny_station=None, deny_station_hours=None):
    """Loads NetCDF inputs matching train/serve pipeline transformations.
    deny_goes_channels selects channels to zero (e.g. ["goes_channel_02","goes_channel_07"]); 
    deny_goes_hours selects hours to deny (0 = most recent, then 1, then 2 (oldest)). Should be provided as a list (e.g. [2]). If omitted, all hours are zeroed.
    deny_station = (y, x) removes that station from the model input only (targets and obs_norm keep it);
    deny_station_hours uses the same hour convention as deny_goes_hours (default: all hours).
    """
    
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    try:
        H = p.img_size_y
        W = p.img_size_x

        lon = np.array(ds.coords["lon"].values)
        lat = np.array(ds.coords["lat"].values)

        # Topo
        topo = ds[["z"]].to_array().to_numpy()

        # Background prediction input (formerly HRRR)
        inp_pred = ds[p.inp_pred_vars].to_array().to_numpy()
        inp_pred = np.squeeze(inp_pred)

        # Observations
        obs = ds[p.inp_obs_vars].to_array().to_numpy()[:, -p.obs_time_window:, :]

        # Satellite Data (4 bands x 3 time steps -> 12 channels)
        sat_window = getattr(p, "obs_time_window", 3)
        sat = ds[p.inp_sat_vars].to_array().to_numpy()[:, -sat_window:, :]

        if deny_goes_channels:
            channel_names = list(p.inp_sat_vars)
            unknown = set(deny_goes_channels) - set(channel_names)
            if unknown:
                raise ValueError(f"Unknown satellite channels: {sorted(unknown)}")

            hours = (
                range(sat.shape[1])
                if deny_goes_hours is None
                else deny_goes_hours
            )
            hours = list(hours)
            if any(not isinstance(hour, int) or hour < 0 or hour >= sat.shape[1]
                for hour in hours):
                raise ValueError(
                    f"deny_goes_hours must contain integers from 0 to {sat.shape[1] - 1}"
                )

            for channel in deny_goes_channels:
                channel_idx = channel_names.index(channel)
                for hour in hours:
                    sat[channel_idx, -(hour + 1), :, :] = 0

        inp_sat = sat.reshape((-1, H, W))

        # Optional METAR filtering from CLI/flag
        if (not include_metar) and ("obs_source" in ds):
            obs_source = ds["obs_source"].to_numpy()
            obs[:, :, obs_source == 2] = 0

        obs_presence_mask = (obs != 0).any(axis=0).astype(np.uint8)
        obs_tar = obs[:, -1]
        obs_tar_mask = (obs_tar != 0).astype(np.float32)

        # Hold-out mask generation matching training dataloader
        if p.hold_out_obs:
            obs_idx = np.flatnonzero((obs[:, -1] != 0).any(axis=0).ravel())
            hold_out_num = int(len(obs_idx) * p.hold_out_obs_ratio)

            if p.obs_mask_seed is None or p.obs_mask_seed < 0:
                rng = np.random.default_rng()
            else:
                digits = "".join(c for c in os.path.basename(nc_file) if c.isdigit())
                rng = np.random.default_rng([int(p.obs_mask_seed), int(digits or 0)])

            hold_out_idx = rng.choice(obs_idx, size=hold_out_num, replace=False)

            obs_mask = np.zeros(H * W, dtype=np.float32)
            obs_mask[hold_out_idx] = 1.0
            obs_mask = obs_mask.reshape(H, W)

            inp_obs = obs * (1.0 - obs_mask)
        else:
            inp_obs = obs.copy()
            obs_mask = np.zeros((H, W), dtype=np.float32)

        if deny_station is not None:
            hours = list(range(obs.shape[1]) if deny_station_hours is None else deny_station_hours)
            if any(not isinstance(hour, int) or hour < 0 or hour >= obs.shape[1] for hour in hours):
                raise ValueError(f"deny_station_hours must contain integers from 0 to {obs.shape[1] - 1}")
            station_y, station_x = deny_station
            for hour in hours:
                inp_obs[:, -(hour + 1), station_y, station_x] = 0
        inp_obs = inp_obs.reshape((-1, H, W))

        # Targets (Normalized space)
        field_tar = ds[p.field_tar_vars].to_array().to_numpy()[:, :H, :W]

        field_obs_tar = field_tar.copy()
        field_obs_tar[obs_tar_mask == 1] = 0
        field_obs_tar += obs_tar

        # Target residual in normalized space if model learns residual
        if p.learn_residual:
            field_tar_res = field_tar - inp_pred
            obs_tar_res = obs_tar - inp_pred
            field_obs_tar_res = field_obs_tar - inp_pred
        else:
            field_tar_res = field_tar
            obs_tar_res = obs_tar
            field_obs_tar_res = field_obs_tar

        # Concatenate in training channel order: [inp_pred, inp_obs, inp_sat, topo]
        inp = np.concatenate((inp_pred, inp_obs, inp_sat, topo), axis=0).astype(np.float32)

        aux = {
            "lat": lat,
            "lon": lon,
            "inp_pred": inp_pred.astype(np.float32),
            "inp_obs": inp_obs.astype(np.float32),
            "inp_sat": inp_sat.astype(np.float32),
            "topo": topo.astype(np.float32),
            "target_field_norm": field_tar.astype(np.float32),
            "target_field_res_norm": field_tar_res.astype(np.float32),
            "target_obs_res_norm": obs_tar_res.astype(np.float32),
            "target_field_obs_res_norm": field_obs_tar_res.astype(np.float32),
            "obs_tar_mask": obs_tar_mask.astype(np.float32),
            "obs_mask": obs_mask.astype(np.float32),
            "obs_presence_mask": obs_presence_mask,
            # station obs before random holdout (after optional METAR removal); 0 = no ob
            "obs_norm": obs.astype(np.float32),
        }
        return inp, aux
    finally:
        ds.close()


def run_model_inference(model, nc_path, params, stats_path, device, include_metar=True, deny_goes_channels=None, deny_goes_hours=None,
                        deny_station=None, deny_station_hours=None):
    """Runs model inference on a NetCDF file, detects fill values and sets outside/padding 
    regions to NaN, unnormalizes outputs correctly, and returns a packaged dictionary of results.
    """
    # 1. Load data
    inp_np, aux = build_model_input_from_netcdf(nc_path, params, include_metar=include_metar, deny_goes_channels=deny_goes_channels, deny_goes_hours=deny_goes_hours,
                                                deny_station=deny_station, deny_station_hours=deny_station_hours)
    inp_tensor = torch.from_numpy(inp_np).unsqueeze(0).to(device)

    # 2. Inference
    model.eval()
    with torch.no_grad():
        pred_tensor = model(inp_tensor)

    pred_norm = pred_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # 3. Load Stats for Unnormalization
    rtma_anl_vmin, rtma_anl_vmax = load_stats(stats_path, params.field_tar_vars)
    pred_input_vmin, pred_input_vmax = load_stats(stats_path, params.inp_pred_vars)

    inp_pred_norm = aux["inp_pred"].copy()

    # 4. Analysis Reconstruction (in Normalized Space)
    if params.learn_residual:
        pred_analysis_norm = pred_norm + inp_pred_norm
        target_analysis_norm = aux["target_field_res_norm"] + inp_pred_norm
    else:
        pred_analysis_norm = pred_norm
        target_analysis_norm = aux["target_field_res_norm"]

    # 5. Reverse Normalization to Physical Units
    inp_pred_unnorm = reverse_norm(inp_pred_norm, pred_input_vmin, pred_input_vmax, channel_axis=0)
    pred_analysis_unnorm = reverse_norm(pred_analysis_norm, rtma_anl_vmin, rtma_anl_vmax, channel_axis=0)
    target_analysis_unnorm = reverse_norm(target_analysis_norm, rtma_anl_vmin, rtma_anl_vmax, channel_axis=0)

    # Physical residual (innovation) = Analysis - Background Prediction
    pred_residual_unnorm = pred_analysis_unnorm - inp_pred_unnorm
    target_residual_unnorm = target_analysis_unnorm - inp_pred_unnorm

    # 6. Channel maps and result packaging
    output_channel_names = [
        f"output_{v.split('rtma_anl_', 1)[1]}" if v.startswith("rtma_anl_") else f"output_{v}"
        for v in params.field_tar_vars
    ]

    if "output_q" in output_channel_names:
        q_idx = output_channel_names.index("output_q")
        for output_array in (
            pred_analysis_unnorm,
            target_analysis_unnorm,
            pred_residual_unnorm,
            target_residual_unnorm,
        ):
            output_array[q_idx] *= 1000.0

    channel_maps = {
        "input_pred": {i: v for i, v in enumerate(params.inp_pred_vars)},
        "input_obs": {i: v for i, v in enumerate(params.inp_obs_vars)},
        "input_sat": {i: v for i, v in enumerate(getattr(params, "inp_sat_vars", []))},
        "output": {i: v for i, v in enumerate(output_channel_names)},
        "target_field": {i: v for i, v in enumerate(params.field_tar_vars)},
    }

    results = {
        # Tensors & Normalized Arrays
        "prediction_tensor": pred_tensor,
        "prediction_array_norm": pred_norm,
        "input_tensor": inp_tensor,
        "input_array_norm": inp_np,
        "target_field_array_norm": aux["target_field_norm"],
        "target_obs_array_norm": aux["target_obs_res_norm"],
        "target_field_obs_array_norm": aux["target_field_obs_res_norm"],

        # Unnormalized Arrays (Physical Units)
        "prediction_residual_unnorm": pred_residual_unnorm,
        "target_residual_unnorm": target_residual_unnorm,
        "inp_pred_unnorm": inp_pred_unnorm,
        "prediction_analysis_unnorm": pred_analysis_unnorm,
        "target_analysis_unnorm": target_analysis_unnorm,

        # Metadata & Coordinates
        "channel_maps": channel_maps,
        "output_channel_names": output_channel_names,
        "obs_tar_mask_array": aux["obs_tar_mask"],
        "heldout_mask": aux["obs_mask"],
        "lat": aux["lat"],
        "lon": aux["lon"],
    }

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_output_channel(results_dict, channel_name, 
                        channel_to_select="prediction_analysis_unnorm", 
                        colorbar_scale_style="normal",
                        abs_min=None, abs_max=None,
                        use_abs_err=False,
                        title_str=None, 
                        colorbar_label=None, 
                        plot_savepath=None, 
                        cmap='bwr'):
    """Plots one unnormalized output channel by variable name (e.g., 'output_t')."""
    output_names = results_dict['output_channel_names']
    if channel_name not in output_names:
        raise KeyError(
            f"Unknown channel '{channel_name}'. Available: {output_names}"
        )

    idx = output_names.index(channel_name)
    arr = results_dict[channel_to_select][idx]

    if use_abs_err:
        arr = np.abs(arr)
    
    lat = np.asarray(results_dict['lat'])
    lon = np.asarray(results_dict['lon'])
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]

    if colorbar_scale_style == "centered":
        abs_val = np.nanmax(np.abs(arr))
        vmin, vmax = -abs_val, abs_val
    elif colorbar_scale_style == 'extreme':
        vmin, vmax = abs_min, abs_max
    elif colorbar_scale_style == "normal":
        vmin, vmax = None, None
    else:
        raise ValueError("colorbar_scale_style must be 'normal', 'extreme', or 'centered'")

    fig, ax = plt.subplots(figsize=(12, 6)) 
    im = ax.imshow(arr, origin='lower', cmap=cmap, extent=extent, aspect='auto', vmin=vmin, vmax=vmax)

    if title_str is None:
        ax.set_title(f"{channel_name} ({channel_to_select}) | min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}")
    else:
        ax.set_title(title_str)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.016)
    if colorbar_label is None:
        cbar.set_label(f"{channel_name} value")
    else:
        cbar.set_label(colorbar_label)

    if plot_savepath is not None:
        plt.savefig(plot_savepath, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()