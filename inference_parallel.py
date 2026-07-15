import argparse
import glob
import os
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr
import hdf5plugin
from torch.utils.data import DataLoader, Dataset

from models.encdec import build_model
from utils.YParams import YParams

#########################
# Run ADAF_RTMA model inference on many NetCDF files at once and write outputs that keep original fields and append model fields.
# Example to run on command line (must be on a compute node with ADAF_environment active!):
# python apply_ckpt_to_netcdf.py \
#     --input_dir /scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/data_blosc_combined/test_data \
#     --output_dir ./test_output \
#     --checkpoint_path /scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/16657218/ckpt.tar \
#     --stats_path /scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation/stats.csv \
#     --glob_pattern '2023-01-0[1-7]_*.nc' \
#     --batch_size 7 \
#     --write_residual_fields \
#     --overwrite
#########################


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_filepath", type=str, default="./config/params_default.yaml")
    parser.add_argument("--config_name", type=str, default="EncDec")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--stats_path", type=str, default="./data_preparation/stats.csv")

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--glob_pattern", type=str, default="*.nc")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--hold_out_obs", type=str, default=None)
    parser.add_argument("--hold_out_obs_ratio", type=float, default=None)
    parser.add_argument("--obs_mask_seed", type=int, default=None)

    parser.add_argument("--exclude_metar", action="store_true")
    parser.add_argument("--write_residual_fields", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def str_to_bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        raise KeyError("Checkpoint does not contain 'model_state' or 'state_dict'.")

    # Strip DDP (`module.`) AND torch.compile (`_orig_mod.`) prefixes in any
    # order/combination -- these checkpoints are compile+DDP, so keys look like
    # `module._orig_mod.<param>`. Stripping only `module.` leaves `_orig_mod.`,
    # which matches nothing => strict=False would SILENTLY load a random-init
    # model. strict=True below turns that into a hard error instead.
    def _strip(k):
        changed = True
        while changed:
            changed = False
            for pre in ("_orig_mod.", "module."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        return k

    clean_state = {_strip(key): val for key, val in state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=True)
    return ckpt, missing, unexpected


def load_stats(stats_path, var_names):
    stats_df = pd.read_csv(stats_path).set_index("variable")
    vmin = np.array([stats_df.loc[v, "min"] for v in var_names], dtype=np.float32)
    vmax = np.array([stats_df.loc[v, "max"] for v in var_names], dtype=np.float32)
    return vmin, vmax


def load_stats_map(stats_path):
    stats_df = pd.read_csv(stats_path).set_index("variable")
    return {
        name: (float(stats_df.loc[name, "min"]), float(stats_df.loc[name, "max"]))
        for name in stats_df.index
    }


def reverse_norm(arr, vmin, vmax, channel_axis=None):
    arr = np.asarray(arr, dtype=np.float32)
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    if channel_axis is None:
        if vmin.size != 1 or vmax.size != 1:
            raise ValueError("Scalar unnormalization requires exactly one min/max value.")
        return (arr + 1.0) * (vmax[0] - vmin[0]) / 2.0 + vmin[0]

    if channel_axis < 0:
        channel_axis = arr.ndim + channel_axis
    if channel_axis < 0 or channel_axis >= arr.ndim:
        raise ValueError(f"Invalid channel_axis={channel_axis} for shape {arr.shape}.")

    if arr.shape[channel_axis] != vmin.size:
        raise ValueError(
            f"Channel mismatch: axis {channel_axis} has {arr.shape[channel_axis]} channels "
            f"but stats has {vmin.size}."
        )

    reshape = [1] * arr.ndim
    reshape[channel_axis] = vmin.size
    vmin_b = vmin.reshape(reshape)
    vmax_b = vmax.reshape(reshape)
    return (arr + 1.0) * (vmax_b - vmin_b) / 2.0 + vmin_b


def build_model_input(file_path, params, include_metar):
    ds = xr.open_dataset(file_path, engine="netcdf4")
    try:
        h = params.img_size_y
        w = params.img_size_x

        # Mirror ADAF_RTMA/utils/dataloader_multifiles.py array construction.
        topo = ds[["z"]].to_array().to_numpy()[:, :h, :w]

        inp_hrrr = ds[params.inp_hrrr_vars].to_array().to_numpy()[:, :h, :w]
        inp_hrrr = np.squeeze(inp_hrrr)

        obs = ds[params.inp_obs_vars].to_array().to_numpy()[
            :, -params.obs_time_window :, :h, :w
        ]

        if (not include_metar) and ("obs_source" in ds):
            obs_source = ds["obs_source"].to_numpy()[:h, :w]
            obs[:, :, obs_source == 2] = 0

        if params.hold_out_obs:
            # Station set = cells reporting ANY variable at ANALYSIS time (obs[:, -1]),
            # matching the training dataloader. The old obs[0,0] (variable 0, OLDEST time
            # bin) held out stations that have no analysis-time ob to score, and never
            # held out stations that only start reporting at analysis time.
            obs_idx = np.flatnonzero((obs[:, -1] != 0).any(axis=0).ravel())
            hold_out_num = int(len(obs_idx) * params.hold_out_obs_ratio)

            # seed None (not 0) means "unseeded": 0 is a perfectly good seed, and the old
            # sentinel silently made every past eval hold-out draw irreproducible.
            # Mix the cycle's own timestamp into the entropy so one seed doesn't draw the
            # same positions out of every file's station list -- reproducible, not repeated.
            if params.obs_mask_seed is None:
                rng = np.random.default_rng()
            else:
                digits = "".join(c for c in os.path.basename(file_path) if c.isdigit())
                rng = np.random.default_rng([int(params.obs_mask_seed), int(digits or 0)])
            hold_out_idx = rng.choice(obs_idx, size=hold_out_num, replace=False)

            obs_mask = np.zeros(obs.shape[-2] * obs.shape[-1], dtype=obs.dtype)
            obs_mask[hold_out_idx] = 1
            obs_mask = obs_mask.reshape(obs.shape[-2], obs.shape[-1])

            inp_obs = obs * (1 - obs_mask)
            inp_obs = inp_obs.reshape((-1, h, w))
        else:
            inp_obs = obs.copy()
            inp_obs = inp_obs.reshape((-1, h, w))
            obs_mask = np.zeros((h, w), dtype=obs.dtype)

        inp = np.concatenate((inp_hrrr, inp_obs, topo), axis=0).astype(np.float32)

        # obs_mask (h,w): 1 at cells whose obs were HELD OUT of the model input
        # (all-zero when hold_out_obs is off). The raw input carries an unrelated
        # obs_mask; the held-out eval reads THIS one, so it is written to output.
        return inp, inp_hrrr.astype(np.float32), obs_mask.astype(np.float32)
    finally:
        ds.close()


class InferenceDataset(Dataset):
    def __init__(self, file_paths, params, include_metar):
        self.file_paths = file_paths
        self.params = params
        self.include_metar = include_metar

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        inp, inp_hrrr, obs_mask = build_model_input(path, self.params, self.include_metar)
        return torch.from_numpy(inp), torch.from_numpy(inp_hrrr), torch.from_numpy(obs_mask), path


def collate_batch(batch):
    inputs = torch.stack([row[0] for row in batch], dim=0)
    hrrr = torch.stack([row[1] for row in batch], dim=0)
    obs_mask = torch.stack([row[2] for row in batch], dim=0)
    paths = [row[3] for row in batch]
    return inputs, hrrr, obs_mask, paths


def build_output_names(field_tar_vars):
    short_names = []
    for name in field_tar_vars:
        if name.startswith("rtma_"):
            short_names.append(name.split("rtma_", 1)[1])
        else:
            short_names.append(name)

    analysis_names = [f"output_{name}" for name in short_names]
    residual_names = [f"output_residual_{name}" for name in short_names]
    return analysis_names, residual_names


def crop_to_shared_domain(ds, pred_analysis, pred_residual):
    ds_y = ds.sizes["y"]
    ds_x = ds.sizes["x"]
    pred_y = pred_analysis.shape[-2]
    pred_x = pred_analysis.shape[-1]

    out_y = min(ds_y, pred_y)
    out_x = min(ds_x, pred_x)

    if ds_y != out_y or ds_x != out_x:
        ds = ds.isel(y=slice(0, out_y), x=slice(0, out_x))

    if pred_y != out_y or pred_x != out_x:
        pred_analysis = pred_analysis[:, :out_y, :out_x]
        pred_residual = pred_residual[:, :out_y, :out_x]

    return ds, pred_analysis, pred_residual


def replace_normalized_inputs_in_place(ds_out, params, stats_map):
    hrrr_vars = [v for v in params.inp_hrrr_vars if v in ds_out and v in stats_map]
    if len(hrrr_vars) > 0:
        hrrr_arr = np.stack([ds_out[v].to_numpy().astype(np.float32) for v in hrrr_vars], axis=0)
        hrrr_vmin = np.array([stats_map[v][0] for v in hrrr_vars], dtype=np.float32)
        hrrr_vmax = np.array([stats_map[v][1] for v in hrrr_vars], dtype=np.float32)
        hrrr_unnorm = reverse_norm(hrrr_arr, hrrr_vmin, hrrr_vmax, channel_axis=0)
        for i, var_name in enumerate(hrrr_vars):
            ref = ds_out[var_name]
            ds_out[var_name] = xr.DataArray(
                hrrr_unnorm[i], dims=ref.dims, coords=ref.coords, attrs=ref.attrs
            )

    obs_vars = [v for v in params.inp_obs_vars if v in ds_out and v in stats_map]
    if len(obs_vars) > 0:
        obs_arr = np.stack([ds_out[v].to_numpy().astype(np.float32) for v in obs_vars], axis=0)
        obs_vmin = np.array([stats_map[v][0] for v in obs_vars], dtype=np.float32)
        obs_vmax = np.array([stats_map[v][1] for v in obs_vars], dtype=np.float32)
        obs_unnorm = reverse_norm(obs_arr, obs_vmin, obs_vmax, channel_axis=0)
        for i, var_name in enumerate(obs_vars):
            ref = ds_out[var_name]
            ds_out[var_name] = xr.DataArray(
                obs_unnorm[i], dims=ref.dims, coords=ref.coords, attrs=ref.attrs
            )

    if "z" in ds_out and "z" in stats_map:
        zmin, zmax = stats_map["z"]
        ref = ds_out["z"]
        z_unnorm = reverse_norm(ref.to_numpy().astype(np.float32), [zmin], [zmax], channel_axis=None)
        ds_out["z"] = xr.DataArray(z_unnorm, dims=ref.dims, coords=ref.coords, attrs=ref.attrs)

    # rtma (field_tar_vars) is the truth analysis and is stored NORMALIZED in the
    # raw input; un-normalize it here so the output file is uniformly in base
    # units (output_/hrrr_/sta_ already are). Without this, error maps
    # (output_<v> - rtma_<v>) and the held-out eval mix normalized+base units.
    tar_vars = [v for v in params.field_tar_vars if v in ds_out and v in stats_map]
    if len(tar_vars) > 0:
        tar_arr = np.stack([ds_out[v].to_numpy().astype(np.float32) for v in tar_vars], axis=0)
        tar_vmin = np.array([stats_map[v][0] for v in tar_vars], dtype=np.float32)
        tar_vmax = np.array([stats_map[v][1] for v in tar_vars], dtype=np.float32)
        tar_unnorm = reverse_norm(tar_arr, tar_vmin, tar_vmax, channel_axis=0)
        for i, var_name in enumerate(tar_vars):
            ref = ds_out[var_name]
            ds_out[var_name] = xr.DataArray(
                tar_unnorm[i], dims=ref.dims, coords=ref.coords, attrs=ref.attrs
            )

    return ds_out


def write_output_file(
    input_file,
    output_file,
    pred_analysis,
    pred_residual,
    obs_mask,
    stats_map,
    params,
    analysis_names,
    residual_names,
    write_residual_fields,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with xr.open_dataset(input_file, engine="netcdf4") as ds_in:
        ds_out, pred_analysis, pred_residual = crop_to_shared_domain(
            ds_in, pred_analysis, pred_residual
        )
        ds_out = replace_normalized_inputs_in_place(ds_out, params, stats_map)
        y_vals = ds_out["y"].values
        x_vals = ds_out["x"].values

        # Held-out mask, cropped to the shared (cropped) domain. Written as a NEW
        # var `heldout_mask` -- NOT the raw input's `obs_mask` (an unrelated
        # data-quality/complete-station field the held-out eval still needs to
        # detect degenerate year-boundary cycles), which is preserved untouched.
        out_y, out_x = len(y_vals), len(x_vals)
        heldout_mask = np.asarray(obs_mask)[:out_y, :out_x].astype(np.int8)

        data_vars = {
            "heldout_mask": xr.DataArray(
                heldout_mask,
                dims=("y", "x"),
                coords={"y": y_vals, "x": x_vals},
                attrs={"long_name": "held-out obs mask (1 = obs withheld from model input)",
                       "hold_out_obs_ratio": float(params.hold_out_obs_ratio or 0.0),
                       # -1 records "unseeded" (seed None); 0 is a real seed, not a sentinel.
                       "obs_mask_seed": -1 if params.obs_mask_seed is None else int(params.obs_mask_seed)},
            )
        }
        for i, var_name in enumerate(analysis_names):
            attrs = {
                "long_name": f"ADAF analysis output for {params.field_tar_vars[i]}",
                "source": "ADAF_RTMA EncDec inference",
                "checkpoint": str(params.checkpoint_path),
            }
            if params.field_tar_vars[i] in ds_out and "units" in ds_out[params.field_tar_vars[i]].attrs:
                attrs["units"] = ds_out[params.field_tar_vars[i]].attrs["units"]

            data_vars[var_name] = xr.DataArray(
                pred_analysis[i].astype(np.float32),
                dims=("y", "x"),
                coords={"y": y_vals, "x": x_vals},
                attrs=attrs,
            )

        if write_residual_fields:
            for i, var_name in enumerate(residual_names):
                attrs = {
                    "long_name": f"ADAF residual output for {params.field_tar_vars[i]}",
                    "source": "ADAF_RTMA EncDec inference",
                    "checkpoint": str(params.checkpoint_path),
                }
                if params.field_tar_vars[i] in ds_out and "units" in ds_out[params.field_tar_vars[i]].attrs:
                    attrs["units"] = ds_out[params.field_tar_vars[i]].attrs["units"]

                data_vars[var_name] = xr.DataArray(
                    pred_residual[i].astype(np.float32),
                    dims=("y", "x"),
                    coords={"y": y_vals, "x": x_vals},
                    attrs=attrs,
                )

        ds_pred = xr.Dataset(data_vars=data_vars)
        ds_merged = xr.merge([ds_out, ds_pred], compat="no_conflicts")
        ds_merged.load()

    # Global provenance, so a plot can label itself without being told which run made it.
    ds_merged.attrs["checkpoint"] = str(params.checkpoint_path)
    epoch = getattr(params, "checkpoint_epoch", None)
    if epoch is not None:
        ds_merged.attrs["checkpoint_epoch"] = int(epoch)

    comp = {"zlib": True, "complevel": 1}
    encoding = {name: comp for name in ds_merged.data_vars}
    ds_merged.to_netcdf(output_file, mode="w", encoding=encoding)


def main():
    t=time.time()
    
    args = get_args()

    params = YParams(args.config_filepath, args.config_name)
    params.override_from_cli(args)

    if args.hold_out_obs is not None:
        params["hold_out_obs"] = str_to_bool(args.hold_out_obs)
    if args.hold_out_obs_ratio is not None:
        params["hold_out_obs_ratio"] = float(args.hold_out_obs_ratio)
    if args.obs_mask_seed is not None:
        # Negative = "unseeded" (None). 0 is a real seed; the old code treated it as the
        # no-seed sentinel, which is why past eval hold-out draws were not reproducible.
        params["obs_mask_seed"] = (None if int(args.obs_mask_seed) < 0
                                   else int(args.obs_mask_seed))

    include_metar = not args.exclude_metar

    if args.checkpoint_path is not None:
        params["checkpoint_path"] = args.checkpoint_path

    if params.checkpoint_path is None:
        raise ValueError("checkpoint_path is not set. Provide --checkpoint_path or set it in config.")

    file_paths = sorted(glob.glob(os.path.join(args.input_dir, args.glob_pattern)))
    if len(file_paths) == 0:
        raise FileNotFoundError(
            f"No NetCDF files found in {args.input_dir} with pattern {args.glob_pattern}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(params).to(device)
    ckpt, missing, unexpected = load_checkpoint(model, params.checkpoint_path, device)
    model.eval()
    params["checkpoint_epoch"] = ckpt.get("epoch")
    print(f"Checkpoint epoch: {params.checkpoint_epoch}")

    if len(missing) > 0:
        print(f"Warning: missing checkpoint keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"Warning: unexpected checkpoint keys: {len(unexpected)}")

    stats_map = load_stats_map(args.stats_path)
    rtma_vmin, rtma_vmax = load_stats(args.stats_path, params.field_tar_vars)
    hrrr_vmin, hrrr_vmax = load_stats(args.stats_path, params.inp_hrrr_vars)

    dataset = InferenceDataset(file_paths, params, include_metar)

    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_batch,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    loader = DataLoader(dataset, **loader_kwargs)

    analysis_names, residual_names = build_output_names(params.field_tar_vars)

    print("----- Inference configuration -----")
    print(f"Device: {device}")
    print(f"Checkpoint: {params.checkpoint_path}")
    print(f"Stats: {args.stats_path}")
    print(f"Input files discovered: {len(file_paths)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data workers: {args.num_workers}")
    print(f"Include METAR: {include_metar}")
    print("-----------------------------------")

    total = len(file_paths)
    processed = 0
    written = 0
    skipped_existing = 0

    with torch.no_grad():
        for batch_inputs, batch_hrrr, batch_obs_mask, batch_paths in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            pred = model(batch_inputs)

            pred_norm = pred.detach().cpu().numpy().astype(np.float32)
            hrrr_norm = batch_hrrr.cpu().numpy().astype(np.float32)
            obs_mask_np = batch_obs_mask.cpu().numpy()

            # Reconstruct the analysis the way the model was trained: the model
            # predicts the residual in NORMALIZED space, so the analysis is
            # (residual_norm + hrrr_norm) and gets ONE reverse-norm to base
            # units. Un-normalizing residual and background SEPARATELY and adding
            # double-counts the min-max offset (VMIN + HALF_RANGE): e.g. +12.5
            # g/kg on q, +5 degC on t. (hrrr_vmin/vmax == rtma_vmin/vmax and the
            # channels are t,q,u10,v10-aligned in both configs, so the norm-space
            # sum is valid.)
            hrrr_unnorm = reverse_norm(hrrr_norm, hrrr_vmin, hrrr_vmax, channel_axis=1)
            if params.learn_residual:
                pred_analysis = reverse_norm(pred_norm + hrrr_norm, rtma_vmin, rtma_vmax,
                                             channel_axis=1)
            else:
                pred_analysis = reverse_norm(pred_norm, rtma_vmin, rtma_vmax, channel_axis=1)
            # Innovation over the background, in base units (analysis - HRRR).
            pred_residual = pred_analysis - hrrr_unnorm

            for i, input_file in enumerate(batch_paths):
                output_file = os.path.join(args.output_dir, os.path.basename(input_file))

                if os.path.exists(output_file) and (not args.overwrite):
                    skipped_existing += 1
                    continue

                write_output_file(
                    input_file=input_file,
                    output_file=output_file,
                    pred_analysis=pred_analysis[i],
                    pred_residual=pred_residual[i],
                    obs_mask=obs_mask_np[i],
                    stats_map=stats_map,
                    params=params,
                    analysis_names=analysis_names,
                    residual_names=residual_names,
                    write_residual_fields=args.write_residual_fields,
                )
                written += 1

            processed += len(batch_paths)
            print(f"Processed {processed}/{total} | written={written} skipped_existing={skipped_existing}")

    print(f"\nRun complete. Time = {time.time()-t:.3f} sec")
    print(f"Total files discovered: {total}")
    print(f"Output files written: {written}")
    print(f"Skipped existing files: {skipped_existing}")
    print(f"Checkpoint top-level keys: {list(ckpt.keys())}")


if __name__ == "__main__":
    main()
