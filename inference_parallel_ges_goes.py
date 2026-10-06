r"""Batched GES/GOES inference with parallel input loading and compact NetCDF outputs.

Output fields use the physical units of stats_ges.csv (no additional humidity scaling).
Innovations are analysis minus background; errors are analysis minus target field.

Command-line example (on a compute node with ADAF_environment already active,
run from the ADAF_RTMA directory; replace the input/output/checkpoint/stats paths):

	python inference_parallel_ges_goes.py \
		--input_dir /path/to/test_data \ #/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes/test_data
		--output_dir /path/to/inference_output \ 
		--checkpoint_path /path/to/ckpt.tar \
		--stats_path /path/to/stats_ges.csv \ #/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation_ges/stats_ges.csv
		--start_time '2023-08-20_06' \
		--end_time '2023-08-21_18' \
		--batch_size 8 \ #8 is the current limit on a 96 GB VRAM H100
		--num_workers 2

Both time limits are inclusive; omit them to process all matching files.
Outputs always include rtma_anl_<var> (target analysis) and obs_<var> (station obs over the
observation time window, NaN where no ob, before random holdout).
With --hold_out_obs True --hold_out_obs_ratio 0.1, outputs also include
obs_mask and held_out_obs over the observation time window.
"""

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import hdf5plugin
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from utils.YParams import YParams
from utils.inference_functions import (
	build_model_input_from_netcdf,
	load_checkpoint_weights,
	load_stats,
	reverse_norm,
)


def str_to_bool(value):
	text = str(value).strip().lower()
	if text in {"1", "true", "t", "yes", "y"}:
		return True
	if text in {"0", "false", "f", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError("Expected true or false.")


def parse_hour(value):
	for format_string in ("%Y-%m-%d_%H", "%Y_%m_%d_%H"):
		try:
			return datetime.strptime(value, format_string)
		except ValueError:
			pass
	raise argparse.ArgumentTypeError(
		f"Invalid hour {value!r}; expected YYYY-MM-DD_HH or YYYY_MM_DD_HH."
	)


def filter_files_by_time(file_paths, start_time=None, end_time=None):
	if start_time is None and end_time is None:
		return file_paths
	selected = []
	for path in file_paths:
		try:
			file_time = parse_hour(Path(path).stem)
		except argparse.ArgumentTypeError as exc:
			raise ValueError(f"Cannot determine sample hour from filename {path!r}: {exc}") from exc
		if start_time is not None and file_time < start_time:
			continue
		if end_time is not None and file_time > end_time:
			continue
		selected.append(path)
	return selected


def get_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--config_filepath", default=str(Path(__file__).parent / "config/params_lowres_ges_goes.yaml")
	)
	parser.add_argument("--config_name", default="EncDec")
	parser.add_argument("--checkpoint_path", default=None)
	parser.add_argument("--stats_path", required=True)
	parser.add_argument("--input_dir", required=True)
	parser.add_argument("--output_dir", required=True)
	parser.add_argument("--glob_pattern", default="*.nc")
	parser.add_argument("--start_time", type=parse_hour, default=None,
						help="Inclusive first sample hour (YYYY-MM-DD_HH or YYYY_MM_DD_HH).")
	parser.add_argument("--end_time", type=parse_hour, default=None,
						help="Inclusive last sample hour (YYYY-MM-DD_HH or YYYY_MM_DD_HH).")
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--prefetch_factor", type=int, default=1)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--hold_out_obs", type=str_to_bool, default=False)
	parser.add_argument("--hold_out_obs_ratio", type=float, default=0.0)
	parser.add_argument("--obs_mask_seed", type=int, default=None)
	parser.add_argument("--exclude_metar", action="store_true")
	parser.add_argument("--deny_goes_channels", nargs="+", default=None)
	parser.add_argument("--deny_goes_hours", nargs="+", type=int, default=None)
	parser.add_argument("--overwrite", action="store_true")
	args = parser.parse_args()
	if args.batch_size < 1 or args.num_workers < 0 or args.prefetch_factor < 1:
		parser.error("batch_size/prefetch_factor must be positive; num_workers must be nonnegative.")
	if args.deny_goes_hours is not None and not args.deny_goes_channels:
		parser.error("--deny_goes_hours requires --deny_goes_channels.")
	if args.start_time is not None and args.end_time is not None and args.start_time > args.end_time:
		parser.error("--start_time must be earlier than or equal to --end_time.")
	return args


class InferenceDataset(Dataset):
	def __init__(self, file_paths, params, obs_stats, include_metar=True,
				 deny_goes_channels=None, deny_goes_hours=None):
		self.file_paths = file_paths
		self.params = params
		self.obs_stats = obs_stats
		self.include_metar = include_metar
		self.deny_goes_channels = deny_goes_channels
		self.deny_goes_hours = deny_goes_hours

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):
		path = self.file_paths[index]
		inputs, aux = build_model_input_from_netcdf(
			path, self.params, include_metar=self.include_metar,
			deny_goes_channels=self.deny_goes_channels,
			deny_goes_hours=self.deny_goes_hours,
		)
		expected_shape = (self.params.in_chans, self.params.img_size_y, self.params.img_size_x)
		if inputs.shape != expected_shape:
			raise ValueError(f"{path}: input shape {inputs.shape}, expected {expected_shape}.")
		obs_norm = aux["obs_norm"]
		obs = reverse_norm(obs_norm, *self.obs_stats, channel_axis=0)
		# a station is present if any variable is nonzero; zeros at present stations are valid obs
		obs[:, aux["obs_presence_mask"] == 0] = np.nan
		extras = {"obs": torch.from_numpy(obs.astype(np.float32))}
		if self.params.hold_out_obs:
			presence = aux["obs_presence_mask"]
			extras["obs_mask"] = torch.from_numpy(presence)
			extras["held_out_obs"] = torch.from_numpy(
				(presence * aux["obs_mask"][None]).astype(np.uint8)
			)
		return (
			torch.from_numpy(inputs),
			torch.from_numpy(aux["inp_pred"]),
			torch.from_numpy(aux["target_field_norm"]),
			path,
			extras,
		)


def collate_batch(batch):
	return (
		torch.stack([row[0] for row in batch]),
		torch.stack([row[1] for row in batch]),
		torch.stack([row[2] for row in batch]),
		[row[3] for row in batch],
		{name: torch.stack([row[4][name] for row in batch]) for name in batch[0][4]},
	)


def reconstruct_outputs(pred_norm, background_norm, target_norm, params, target_stats, background_stats):
	if pred_norm.shape != background_norm.shape or pred_norm.shape != target_norm.shape:
		raise ValueError(
			f"Prediction, background, and target shapes must match: "
			f"{pred_norm.shape}, {background_norm.shape}, {target_norm.shape}."
		)
	analysis_norm = pred_norm + background_norm if params.learn_residual else pred_norm
	analysis = reverse_norm(analysis_norm, *target_stats, channel_axis=1)
	background = reverse_norm(background_norm, *background_stats, channel_axis=1)
	target = reverse_norm(target_norm, *target_stats, channel_axis=1)
	return analysis, analysis - background, analysis - target, target


def build_output_names(field_tar_vars):
	short_names = [
		name.removeprefix("rtma_anl_") for name in field_tar_vars
	]
	if len(short_names) != 4 or len(set(short_names)) != 4:
		raise ValueError("Expected four distinct target fields.")
	return (
		[f"output_{name}" for name in short_names],
		[f"output_residual_{name}" for name in short_names],
		[f"output_error_{name}" for name in short_names],
	)


def write_output_file(input_file, output_file, analysis, innovation, error, target, params, extras):
	names = build_output_names(params.field_tar_vars)
	height, width = analysis.shape[-2:]
	with xr.open_dataset(input_file, engine="netcdf4") as ds_in:
		if height > ds_in.sizes["y"] or width > ds_in.sizes["x"]:
			raise ValueError(f"Prediction domain exceeds the input domain for {input_file}.")
		cropped = ds_in.isel(y=slice(0, height), x=slice(0, width))
		coords = {
			name: coord.load().copy(deep=True)
			for name, coord in cropped.coords.items()
			if set(coord.dims).issubset({"y", "x"})
		}
		data_vars = {}
		descriptions = ("ADAF analysis", "ADAF analysis minus background", "ADAF analysis minus target")
		for array, field_names, description in zip((analysis, innovation, error), names, descriptions):
			for channel_idx, name in enumerate(field_names):
				target_name = params.field_tar_vars[channel_idx]
				attrs = {"long_name": f"{description} for {target_name}"}
				if "units" in ds_in[target_name].attrs:
					attrs["units"] = ds_in[target_name].attrs["units"]
				data_vars[name] = xr.DataArray(
					array[channel_idx].astype(np.float32), dims=("y", "x"), attrs=attrs
				)
		for channel_idx, target_name in enumerate(params.field_tar_vars):
			data_vars[target_name] = xr.DataArray(
				target[channel_idx].astype(np.float32), dims=("y", "x"),
				attrs={"long_name": f"Target analysis {target_name} (unnormalized)"},
			)

		obs_dims = ds_in[params.inp_obs_vars[0]].dims
		time_dims = [dim for dim in obs_dims if dim not in {"y", "x"}]
		if len(time_dims) != 1:
			raise ValueError(f"Expected one observation time dimension, got {obs_dims}.")
		time_dim = time_dims[0]
		if time_dim in ds_in.coords:
			coords[time_dim] = ds_in[time_dim].isel(
				{time_dim: slice(-params.obs_time_window, None)}
			).load().copy(deep=True)
		for channel_idx, obs_name in enumerate(params.inp_obs_vars):
			data_vars[f"obs_{obs_name.removeprefix('sta_')}"] = xr.DataArray(
				extras["obs"][channel_idx, :, :height, :width].astype(np.float32),
				dims=(time_dim, "y", "x"),
				attrs={"long_name": f"Station observation {obs_name} (unnormalized, NaN where absent, "
									 "before random holdout)"},
			)
		if "obs_mask" in extras:
			mask_descriptions = {
				"obs_mask": "Active station locations before random observation holdout",
				"held_out_obs": "Active station observations denied to the model by random holdout",
			}
			for name, description in mask_descriptions.items():
				data_vars[name] = xr.DataArray(
					extras[name][:, :height, :width].astype(np.uint8),
					dims=(time_dim, "y", "x"), attrs={
						"long_name": description,
						"flag_values": np.array([0, 1], dtype=np.uint8),
						"flag_meanings": "absent present",
					},
				)
		ds_out = xr.Dataset(data_vars, coords=coords, attrs={
			"source": "ADAF_RTMA GES/GOES EncDec inference",
			"checkpoint": str(params.checkpoint_path),
			"input_file": os.path.basename(input_file),
			"innovation_definition": "analysis - background prediction",
			"error_definition": "analysis - target field (without station substitution)",
			"normalization_units": "Physical units from stats_ges.csv; no additional humidity scaling",
		})
	os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
	blosc2_encoding = dict(hdf5plugin.Blosc2(
		cname="zstd", clevel=3, filters=hdf5plugin.Blosc2.BITSHUFFLE
	))
	encoding = {}
	for name in ds_out.data_vars:
		encoding[name] = blosc2_encoding.copy()
		# one chunk per 2-D field: downstream readers always load whole fields
		encoding[name]["chunksizes"] = tuple(
			1 if dim not in {"y", "x"} else size for dim, size in ds_out[name].sizes.items()
		)
		if ds_out[name].dtype == np.float64:
			encoding[name]["dtype"] = "float32"
	ds_out.to_netcdf(output_file, engine="h5netcdf", mode="w", encoding=encoding)
	ds_out.close()


def main():
	start = time.perf_counter()
	args = get_args()
	params = YParams(args.config_filepath, args.config_name)
	params.override_from_cli(args)
	if not params.checkpoint_path:
		raise ValueError("Provide --checkpoint_path or set checkpoint_path in the config.")
	if not 0 <= float(params.hold_out_obs_ratio) <= 1:
		raise ValueError("hold_out_obs_ratio must be between 0 and 1.")
	build_output_names(params.field_tar_vars)
	if len(params.inp_pred_vars) != 4 or params.out_chans != 4:
		raise ValueError("Expected four background channels and four model outputs.")
	if os.path.realpath(args.input_dir) == os.path.realpath(args.output_dir):
		raise ValueError("input_dir and output_dir must differ to protect input files.")

	discovered = sorted(glob.glob(os.path.join(args.input_dir, args.glob_pattern)))
	if not discovered:
		raise FileNotFoundError(f"No input files match {args.glob_pattern!r} in {args.input_dir}.")
	if args.start_time is not None or args.end_time is not None:
		total_discovered = len(discovered)
		discovered = filter_files_by_time(discovered, args.start_time, args.end_time)
		print(f"Time filter: start={args.start_time}, end={args.end_time}; "
			  f"selected={len(discovered)}/{total_discovered}")
		if not discovered:
			raise FileNotFoundError("No input files fall within the requested time limits.")
	file_paths = [
		path for path in discovered
		if args.overwrite or not os.path.exists(os.path.join(args.output_dir, os.path.basename(path)))
	]
	skipped = len(discovered) - len(file_paths)
	print(f"Discovered {len(discovered)} files; skipped existing={skipped}; queued={len(file_paths)}")
	if not file_paths:
		return

	device = torch.device(args.device)
	if params.arch == "lowres":
		from models.encdec_lowres import LowResEncDec as Model
	else:
		from models.encdec import EncDec as Model
	model = Model(params).to(device)
	_, missing, unexpected = load_checkpoint_weights(model, params.checkpoint_path, device)
	missing_weights = [key for key in missing if "attn_mask" not in key]
	if missing_weights or unexpected:
		raise RuntimeError(f"Checkpoint mismatch: missing={missing_weights}, unexpected={unexpected}")
	model.eval()
	target_stats = load_stats(args.stats_path, params.field_tar_vars)
	background_stats = load_stats(args.stats_path, params.inp_pred_vars)
	obs_stats = load_stats(args.stats_path, params.inp_obs_vars)
	dataset = InferenceDataset(
		file_paths, params, obs_stats, include_metar=not args.exclude_metar,
		deny_goes_channels=args.deny_goes_channels, deny_goes_hours=args.deny_goes_hours,
	)
	loader_kwargs = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"shuffle": False,
		"pin_memory": device.type == "cuda",
		"collate_fn": collate_batch,
	}
	if args.num_workers > 0:
		loader_kwargs["prefetch_factor"] = args.prefetch_factor
		loader_kwargs["multiprocessing_context"] = "spawn"
	loader = DataLoader(dataset, **loader_kwargs)
	print(f"Device={device}; batch_size={args.batch_size}; data workers={args.num_workers}")
	print(f"Checkpoint={params.checkpoint_path}; stats={args.stats_path}")
	written = 0
	with torch.inference_mode():
		for inputs, background, target, paths, extras in loader:
			pred_norm = model(inputs.to(device, non_blocking=True)).float().cpu().numpy()
			analysis, innovation, error, target_phys = reconstruct_outputs(
				pred_norm, background.numpy(), target.numpy(), params, target_stats, background_stats
			)
			for sample_idx, input_file in enumerate(paths):
				output_file = os.path.join(args.output_dir, os.path.basename(input_file))
				write_output_file(
					input_file, output_file, analysis[sample_idx], innovation[sample_idx],
					error[sample_idx], target_phys[sample_idx], params,
					{name: values[sample_idx].numpy() for name, values in extras.items()},
				)
				written += 1
			print(f"Written {written}/{len(file_paths)} | skipped existing={skipped}")
	print(f"Run complete: written={written}, skipped={skipped}, elapsed={time.perf_counter() - start:.3f} sec")


if __name__ == "__main__":
	main()
