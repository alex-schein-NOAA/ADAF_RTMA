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

from models.encdec import EncDec
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

    clean_state = {}
    for key, val in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = val

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
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
            obs_flat = obs[0, 0].flatten()
            obs_idx = np.where(obs_flat != 0)[0]
            hold_out_num = int(len(obs_idx) * params.hold_out_obs_ratio)

            if params.obs_mask_seed != 0:
                np.random.seed(params.obs_mask_seed)

            np.random.shuffle(obs_idx)
            hold_out_idx = obs_idx[:hold_out_num]

            obs_mask = np.zeros(np.shape(obs_flat), dtype=obs.dtype)
            obs_mask[hold_out_idx] = 1
            obs_mask = obs_mask.reshape(obs[0, 0].shape[0], obs[0, 0].shape[1])

            inp_obs = obs * (1 - obs_mask)
            inp_obs = inp_obs.reshape((-1, h, w))
        else:
            inp_obs = obs.copy()
            inp_obs = inp_obs.reshape((-1, h, w))

        inp = np.concatenate((inp_hrrr, inp_obs, topo), axis=0).astype(np.float32)

        return inp, inp_hrrr.astype(np.float32)
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
        inp, inp_hrrr = build_model_input(path, self.params, self.include_metar)
        return torch.from_numpy(inp), torch.from_numpy(inp_hrrr), path


def collate_batch(batch):
    inputs = torch.stack([row[0] for row in batch], dim=0)
    hrrr = torch.stack([row[1] for row in batch], dim=0)
    paths = [row[2] for row in batch]
    return inputs, hrrr, paths


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

    return ds_out


def write_output_file(
    input_file,
    output_file,
    pred_analysis,
    pred_residual,
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

        data_vars = {}
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
        params["obs_mask_seed"] = int(args.obs_mask_seed)

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
    model = EncDec(params).to(device)
    ckpt, missing, unexpected = load_checkpoint(model, params.checkpoint_path, device)
    model.eval()

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
        for batch_inputs, batch_hrrr, batch_paths in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            pred = model(batch_inputs)

            pred_norm = pred.detach().cpu().numpy().astype(np.float32)
            hrrr_norm = batch_hrrr.cpu().numpy().astype(np.float32)

            pred_residual = reverse_norm(pred_norm, rtma_vmin, rtma_vmax, channel_axis=1)
            hrrr_unnorm = reverse_norm(hrrr_norm, hrrr_vmin, hrrr_vmax, channel_axis=1)

            if params.learn_residual:
                pred_analysis = pred_residual + hrrr_unnorm
            else:
                pred_analysis = pred_residual

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
