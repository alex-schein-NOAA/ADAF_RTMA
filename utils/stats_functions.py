"""Statistical evaluation of ADAF inference output (one NetCDF file per hour, named YYYY-MM-DD_HH.nc)."""

import glob
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import multiprocessing as mp

import h5py
import hdf5plugin  # noqa: F401  registers Blosc2 filter used by the inference output
import numpy as np
import xarray as xr

DEFAULT_DATA_ROOT = "/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/inference_output"
VARIABLES = ("t", "q", "u10", "v10")
UNITS = {"t": "K", "q": "kg kg-1", "u10": "m s-1", "v10": "m s-1"}
TIME_FORMAT = "%Y-%m-%d_%H"


def list_model_files(model_dir, start_time=None, end_time=None):
	"""Return sorted (datetime, path) pairs for hourly files within the inclusive time limits."""
	entries = []
	for path in glob.glob(os.path.join(model_dir, "*.nc")):
		try:
			file_time = datetime.strptime(os.path.basename(path)[:-3], TIME_FORMAT)
		except ValueError:
			continue
		if start_time is not None and file_time < start_time:
			continue
		if end_time is not None and file_time > end_time:
			continue
		entries.append((file_time, path))
	entries.sort()
	return entries


def model_file_path(model_dir, file_time):
	return os.path.join(model_dir, f"{file_time.strftime(TIME_FORMAT)}.nc")


def restrict_to_common_times(model_dirs, start_time=None, end_time=None):
	"""Return (sorted times present in every model dir, {model_dir: set of its times})."""
	model_times = {
		model_dir: {entry[0] for entry in list_model_files(model_dir, start_time, end_time)}
		for model_dir in model_dirs
	}
	for model_dir, times in model_times.items():
		if not times:
			raise FileNotFoundError(f"No hourly files found in {model_dir} for the requested time range.")
	common = sorted(set.intersection(*model_times.values()))
	if not common:
		raise ValueError("The requested models share no common output times.")
	excluded = {d: len(t) - len(common) for d, t in model_times.items() if len(t) > len(common)}
	if excluded:
		print(f"NOTE: temporal domain restricted to the {len(common)} times present in all models "
			  f"({common[0].strftime(TIME_FORMAT)} to {common[-1].strftime(TIME_FORMAT)}).")
		for model_dir, n_excluded in excluded.items():
			print(f"  {os.path.basename(os.path.normpath(model_dir))}: {n_excluded} file(s) excluded")
	else:
		print(f"All models share the same {len(common)} times "
			  f"({common[0].strftime(TIME_FORMAT)} to {common[-1].strftime(TIME_FORMAT)}).")
	return common, model_times


def write_missing_times_log(common_times, model_times, output_dir):
	"""Write missing_times_[start]_[end].txt listing hours inside the working domain absent from the common set."""
	start, end = common_times[0], common_times[-1]
	present = set(common_times)
	n_hours = int((end - start) / timedelta(hours=1)) + 1
	missing = [t for t in (start + timedelta(hours=h) for h in range(n_hours)) if t not in present]
	os.makedirs(output_dir, exist_ok=True)
	path = os.path.join(
		output_dir, f"missing_times_{start.strftime(TIME_FORMAT)}_{end.strftime(TIME_FORMAT)}.txt"
	)
	names = {d: os.path.basename(os.path.normpath(d)) for d in model_times}
	with open(path, "w") as log:
		log.write(f"# Working temporal domain: {start.strftime(TIME_FORMAT)} to {end.strftime(TIME_FORMAT)}\n")
		log.write(f"# Hours in domain: {n_hours}; used: {len(common_times)}; missing: {len(missing)}\n")
		log.write(f"# Models: {', '.join(names.values())}\n")
		log.write("# time\tmodels_missing_file\n")
		for t in missing:
			absent = [names[d] for d, times in model_times.items() if t not in times]
			log.write(f"{t.strftime(TIME_FORMAT)}\t{'ALL' if len(absent) == len(names) else ','.join(absent)}\n")
	print(f"Missing times in working domain: {len(missing)}; log written to {path}")
	return path


def _accumulate_chunk(paths, variables):
	"""Per-file domain RMSE plus pixelwise |error| sums and valid counts for a chunk of files."""
	rmse = np.full((len(paths), len(variables)), np.nan, dtype=np.float64)
	abs_sum = None
	count = None
	for file_idx, path in enumerate(paths):
		with h5py.File(path, "r") as h5:
			for var_idx, var in enumerate(variables):
				name = f"output_error_{var}"
				if name not in h5:
					raise KeyError(f"{path} is missing {name}.")
				err = h5[name][...]
				if abs_sum is None:
					abs_sum = np.zeros((len(variables), *err.shape), dtype=np.float64)
					count = np.zeros((len(variables), *err.shape), dtype=np.int32)
				elif err.shape != abs_sum.shape[1:]:
					raise ValueError(f"{path}: {name} shape {err.shape} != {abs_sum.shape[1:]}.")
				valid = np.isfinite(err)
				err = np.where(valid, err, 0.0).astype(np.float64)
				n_valid = valid.sum()
				if n_valid:
					rmse[file_idx, var_idx] = np.sqrt(np.square(err).sum() / n_valid)
				abs_sum[var_idx] += np.abs(err)
				count[var_idx] += valid
	return rmse, abs_sum, count


def _read_latlon(path):
	with h5py.File(path, "r") as h5:
		if "lat" in h5 and "lon" in h5:
			return h5["lat"][...], h5["lon"][...]
	return None, None


def compute_model_stats(model_dir, times, variables=VARIABLES, num_workers=1, files_per_task=None):
	"""Compute domain-average RMSE time series and pixelwise MAE for one model over the given times."""
	paths = [model_file_path(model_dir, t) for t in times]
	if files_per_task is None:
		# each task returns full-domain accumulators, so keep the task count small
		files_per_task = max(1, -(-len(paths) // (4 * max(num_workers, 1))))
	chunks = [paths[i:i + files_per_task] for i in range(0, len(paths), files_per_task)]

	rmse_parts, abs_sum, count = [], None, None
	def reduce(result):
		nonlocal abs_sum, count
		rmse_chunk, abs_chunk, count_chunk = result
		rmse_parts.append(rmse_chunk)
		if abs_sum is None:
			abs_sum, count = abs_chunk, count_chunk
		else:
			abs_sum += abs_chunk
			count += count_chunk

	if num_workers > 1:
		with ProcessPoolExecutor(num_workers, mp_context=mp.get_context("spawn")) as pool:
			# map preserves chunk order, keeping RMSE rows aligned with times
			for done, result in enumerate(pool.map(_accumulate_chunk, chunks, [variables] * len(chunks)), 1):
				reduce(result)
				print(f"  {os.path.basename(model_dir)}: {min(done * files_per_task, len(paths))}/{len(paths)} files", flush=True)
	else:
		for done, chunk in enumerate(chunks, 1):
			reduce(_accumulate_chunk(chunk, variables))
			print(f"  {os.path.basename(model_dir)}: {min(done * files_per_task, len(paths))}/{len(paths)} files", flush=True)

	rmse = np.concatenate(rmse_parts, axis=0)
	with np.errstate(invalid="ignore", divide="ignore"):
		mae = np.where(count > 0, abs_sum / np.maximum(count, 1), np.nan).astype(np.float32)

	data_vars = {}
	for var_idx, var in enumerate(variables):
		units = UNITS.get(var, "")
		data_vars[f"rmse_{var}"] = xr.DataArray(
			rmse[:, var_idx].astype(np.float32), dims=("time",),
			attrs={"long_name": f"Domain-average RMSE of output_error_{var}", "units": units},
		)
		data_vars[f"mae_{var}"] = xr.DataArray(
			mae[var_idx], dims=("y", "x"),
			attrs={"long_name": f"Pixelwise mean absolute error of output_error_{var}", "units": units},
		)
		data_vars[f"n_samples_{var}"] = xr.DataArray(
			count[var_idx], dims=("y", "x"),
			attrs={"long_name": f"Number of valid samples per pixel for {var}"},
		)
	coords = {"time": np.array(times, dtype="datetime64[ns]")}
	lat, lon = _read_latlon(paths[0])
	if lat is not None and lat.shape == mae.shape[1:]:
		coords["lat"] = (("y", "x"), lat.astype(np.float32))
		coords["lon"] = (("y", "x"), lon.astype(np.float32))
	return xr.Dataset(data_vars, coords=coords, attrs={
		"model_dir": os.path.abspath(model_dir),
		"error_definition": "analysis - target field (output_error_* from inference files)",
		"start_time": times[0].strftime(TIME_FORMAT),
		"end_time": times[-1].strftime(TIME_FORMAT),
		"n_files": len(paths),
	})


def stats_filename(model_name, start_time, end_time):
	return f"{model_name}_{start_time.strftime(TIME_FORMAT)}_{end_time.strftime(TIME_FORMAT)}.nc"


def get_or_compute_model_stats(model_dir, times, output_dir, overwrite=False, **kwargs):
	"""Load cached stats for a model if they match the requested times, otherwise compute and save them."""
	model_name = os.path.basename(os.path.normpath(model_dir))
	out_path = os.path.join(output_dir, stats_filename(model_name, times[0], times[-1]))
	if os.path.exists(out_path) and not overwrite:
		cached = xr.load_dataset(out_path)
		same_times = np.array_equal(cached.time.values, np.array(times, dtype="datetime64[ns]"))
		if same_times and all(f"rmse_{var}" in cached for var in kwargs.get("variables", VARIABLES)):
			print(f"Loading cached stats: {out_path}")
			return model_name, cached
	print(f"Computing stats for {model_name} ({len(times)} files)")
	ds = compute_model_stats(model_dir, times, **kwargs)
	save_stats(ds, out_path)
	print(f"Saved {out_path}")
	return model_name, ds


def save_stats(ds, out_path):
	os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
	encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
	ds.to_netcdf(out_path, engine="netcdf4", mode="w", encoding=encoding)


def plot_rmse_timeseries(model_stats, variables, output_dir):
	"""One PNG per variable with a domain-average RMSE line for each model."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import matplotlib.dates as mdates

	starts = [ds.time.values.min() for ds in model_stats.values()]
	ends = [ds.time.values.max() for ds in model_stats.values()]
	tag = (f"{np.datetime_as_string(min(starts), unit='h').replace('T', '_')}_"
		   f"{np.datetime_as_string(max(ends), unit='h').replace('T', '_')}")
	saved = []
	for var in variables:
		fig, ax = plt.subplots(figsize=(12, 4.5))
		for model_name, ds in model_stats.items():
			# hourly reindex inserts NaNs so missing hours show as gaps instead of interpolated lines
			series = ds[f"rmse_{var}"].resample(time="1h").asfreq()
			ax.plot(series.time.values, series.values, lw=1.2, label=model_name)
		ax.set_title(f"Domain-average RMSE: {var}")
		ax.set_xlabel("Valid time (UTC)")
		ax.set_ylabel(f"RMSE [{UNITS.get(var, '')}]")
		locator = mdates.AutoDateLocator()
		ax.xaxis.set_major_locator(locator)
		ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
		ax.grid(alpha=0.3)
		ax.legend(fontsize=8)
		fig.tight_layout()
		path = os.path.join(output_dir, f"rmse_timeseries_{var}_{tag}.png")
		fig.savefig(path, dpi=150)
		plt.close(fig)
		saved.append(path)
	return saved
