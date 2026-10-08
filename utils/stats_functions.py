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
HOLDOUT_SUFFIX = "_hold_out_obs"


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
	"""Per-file domain RMSE plus pixelwise |error| and signed error sums for a chunk of files."""
	rmse = np.full((len(paths), len(variables)), np.nan, dtype=np.float64)
	abs_sum = err_sum = None
	for file_idx, path in enumerate(paths):
		with h5py.File(path, "r") as h5:
			for var_idx, var in enumerate(variables):
				name = f"output_error_{var}"
				if name not in h5:
					raise KeyError(f"{path} is missing {name}.")
				err = h5[name][...].astype(np.float32, copy=False)
				if abs_sum is None:
					abs_sum = np.zeros((len(variables), *err.shape), dtype=np.float32)
					err_sum = np.zeros_like(abs_sum)
				elif err.shape != abs_sum.shape[1:]:
					raise ValueError(f"{path}: {name} shape {err.shape} != {abs_sum.shape[1:]}.")
				if not np.isfinite(err).all():
					raise ValueError(
						f"{path}: {name} has {err.size - np.isfinite(err).sum()} non-finite values; "
						"model output is expected to be complete."
					)
				rmse[file_idx, var_idx] = np.sqrt(np.square(err).mean())
				abs_sum[var_idx] += np.abs(err)
				err_sum[var_idx] += err
	return rmse, abs_sum, err_sum


def _read_latlon(path):
	with h5py.File(path, "r") as h5:
		if "lat" in h5 and "lon" in h5:
			return h5["lat"][...], h5["lon"][...]
	return None, None


def compute_model_stats(model_dir, times, variables=VARIABLES, num_workers=1, files_per_task=None):
	"""Compute domain-average RMSE time series and pixelwise MAE and bias for one model over the given times."""
	paths = [model_file_path(model_dir, t) for t in times]
	if files_per_task is None:
		# each task returns full-domain accumulators, so keep the task count small
		files_per_task = max(1, -(-len(paths) // (4 * max(num_workers, 1))))
	chunks = [paths[i:i + files_per_task] for i in range(0, len(paths), files_per_task)]

	rmse_parts, abs_sum, err_sum = [], None, None
	def reduce(result):
		nonlocal abs_sum, err_sum
		rmse_chunk, abs_chunk, err_chunk = result
		rmse_parts.append(rmse_chunk)
		if abs_sum is None:
			abs_sum, err_sum = abs_chunk, err_chunk
		else:
			abs_sum += abs_chunk
			err_sum += err_chunk

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
	mae = abs_sum / np.float32(len(paths))
	bias = err_sum / np.float32(len(paths))

	data_vars = {
		"n_samples": xr.DataArray(
			np.int32(len(paths)), attrs={"long_name": "Number of hourly samples used at every pixel"}
		),
	}
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
		data_vars[f"bias_{var}"] = xr.DataArray(
			bias[var_idx], dims=("y", "x"),
			attrs={"long_name": f"Pixelwise mean (bias) of output_error_{var}", "units": units},
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
		required = [f"{stat}_{var}" for var in kwargs.get("variables", VARIABLES) for stat in ("rmse", "mae", "bias")]
		if same_times and "n_samples" in cached and all(name in cached for name in required):
			print(f"Loading cached stats: {out_path}")
			return model_name, cached
	print(f"Computing stats for {model_name} ({len(times)} files)")
	ds = compute_model_stats(model_dir, times, **kwargs)
	save_stats(ds, out_path)
	print(f"Saved {out_path}")
	return model_name, ds


def save_stats(ds, out_path):
	os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
	encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars if ds[name].ndim > 0}
	ds.to_netcdf(out_path, engine="netcdf4", mode="w", encoding=encoding)


def _rmse(a, b):
	return np.sqrt(np.mean(np.square(a.astype(np.float64) - b)))


def _heldout_chunk(times, model_dirs, holdout_dirs, variables):
	"""Held-out station RMSE at the analysis hour for every model over a chunk of times."""
	n_t, n_m, n_v = len(times), len(model_dirs), len(variables)
	rmse_obs = np.full((n_t, n_m, n_v), np.nan)
	rmse_anl = np.full((n_t, n_m, n_v), np.nan)
	rmse_ref = np.full((n_t, n_v), np.nan)
	n_stations = np.zeros(n_t, dtype=np.int32)
	n_excluded = np.zeros(n_t, dtype=np.int32)
	for t_idx, file_time in enumerate(times):
		common = union = None
		for holdout_dir in holdout_dirs:
			path = model_file_path(holdout_dir, file_time)
			with h5py.File(path, "r") as h5:
				if "held_out_obs" not in h5:
					raise KeyError(f"{path} is missing held_out_obs.")
				# last obs_time_window entry is the analysis hour
				held = h5["held_out_obs"][-1].astype(bool)
			common = held if common is None else common & held
			union = held if union is None else union | held
		idx = np.flatnonzero(common)
		n_stations[t_idx] = idx.size
		n_excluded[t_idx] = np.count_nonzero(union) - idx.size
		if idx.size == 0:
			continue
		for m_idx, model_dir in enumerate(model_dirs):
			path = model_file_path(model_dir, file_time)
			with h5py.File(path, "r") as h5:
				for v_idx, var in enumerate(variables):
					for name in (f"output_{var}", f"output_error_{var}", f"obs_{var}", f"rtma_anl_{var}"):
						if name not in h5:
							raise KeyError(f"{path} is missing {name}.")
					obs = h5[f"obs_{var}"][-1].ravel()[idx]
					if not np.isfinite(obs).all():
						raise ValueError(
							f"{path}: obs_{var} is missing at {np.count_nonzero(~np.isfinite(obs))} "
							"held-out station(s); were the models run with different --exclude_metar settings?"
						)
					output = h5[f"output_{var}"][...].ravel()[idx]
					error = h5[f"output_error_{var}"][...].ravel()[idx]
					rmse_obs[t_idx, m_idx, v_idx] = _rmse(output, obs)
					rmse_anl[t_idx, m_idx, v_idx] = np.sqrt(np.mean(np.square(error.astype(np.float64))))
					if m_idx == 0:
						rmse_ref[t_idx, v_idx] = _rmse(h5[f"rtma_anl_{var}"][...].ravel()[idx], obs)
	return rmse_obs, rmse_anl, rmse_ref, n_stations, n_excluded


def compute_heldout_stats(model_dirs, times, variables=VARIABLES, num_workers=1):
	"""RMSE time series at held-out stations for all models, vs station obs and vs RTMA analysis.

	Stations are those held out in every *_hold_out_obs model at that hour, so all models are
	scored on the same locations and none of the holdout models saw them.
	"""
	holdout_dirs = [d for d in model_dirs if os.path.normpath(d).endswith(HOLDOUT_SUFFIX)]
	if not holdout_dirs:
		raise ValueError(f"No model directory ends with {HOLDOUT_SUFFIX}.")
	times_per_task = max(1, -(-len(times) // (4 * max(num_workers, 1))))
	chunks = [times[i:i + times_per_task] for i in range(0, len(times), times_per_task)]
	args = [(chunk, model_dirs, holdout_dirs, variables) for chunk in chunks]
	parts = []
	if num_workers > 1:
		with ProcessPoolExecutor(num_workers, mp_context=mp.get_context("spawn")) as pool:
			for done, result in enumerate(pool.map(_heldout_chunk, *zip(*args)), 1):
				parts.append(result)
				print(f"  held-out stations: {min(done * times_per_task, len(times))}/{len(times)} times", flush=True)
	else:
		for done, arg in enumerate(args, 1):
			parts.append(_heldout_chunk(*arg))
			print(f"  held-out stations: {min(done * times_per_task, len(times))}/{len(times)} times", flush=True)
	rmse_obs, rmse_anl, rmse_ref, n_stations, n_excluded = (
		np.concatenate([part[i] for part in parts], axis=0) for i in range(5)
	)

	if n_excluded.any():
		print(f"NOTE: held-out stations differ between holdout models at {np.count_nonzero(n_excluded)} "
			  f"time(s); {int(n_excluded.sum())} station-hours not held out in every model were excluded.")
	if (n_stations == 0).any():
		print(f"WARNING: {np.count_nonzero(n_stations == 0)} time(s) have no common held-out stations (RMSE is NaN).")

	model_names = [os.path.basename(os.path.normpath(d)) for d in model_dirs]
	data_vars = {
		"n_heldout": xr.DataArray(n_stations, dims=("time",), attrs={
			"long_name": "Number of stations held out in every holdout model at the analysis hour"}),
		"n_heldout_excluded": xr.DataArray(n_excluded, dims=("time",), attrs={
			"long_name": "Held-out stations dropped because they were not held out in every holdout model"}),
	}
	for v_idx, var in enumerate(variables):
		units = UNITS.get(var, "")
		data_vars[f"rmse_obs_{var}"] = xr.DataArray(
			rmse_obs[:, :, v_idx].T.astype(np.float32), dims=("model", "time"),
			attrs={"long_name": f"RMSE of output_{var} vs obs_{var} at held-out stations", "units": units},
		)
		data_vars[f"rmse_anl_{var}"] = xr.DataArray(
			rmse_anl[:, :, v_idx].T.astype(np.float32), dims=("model", "time"),
			attrs={"long_name": f"RMSE of output_{var} vs rtma_anl_{var} at held-out stations", "units": units},
		)
		data_vars[f"rmse_rtma_anl_obs_{var}"] = xr.DataArray(
			rmse_ref[:, v_idx].astype(np.float32), dims=("time",),
			attrs={"long_name": f"RMSE of rtma_anl_{var} vs obs_{var} at held-out stations", "units": units},
		)
	return xr.Dataset(
		data_vars,
		coords={"model": np.array(model_names, dtype=object), "time": np.array(times, dtype="datetime64[ns]")},
		attrs={
			"holdout_models": ", ".join(os.path.basename(os.path.normpath(d)) for d in holdout_dirs),
			"station_definition": "held_out_obs at the analysis hour (last obs_time_window entry), "
								  "intersected across holdout models",
			"start_time": times[0].strftime(TIME_FORMAT),
			"end_time": times[-1].strftime(TIME_FORMAT),
		},
	)


def get_or_compute_heldout_stats(model_dirs, times, output_dir, overwrite=False, **kwargs):
	"""Load heldout_stats_[start]_[end].nc if it matches the requested models and times, otherwise recompute."""
	out_path = os.path.join(output_dir, stats_filename("heldout_stats", times[0], times[-1]))
	model_names = [os.path.basename(os.path.normpath(d)) for d in model_dirs]
	if os.path.exists(out_path) and not overwrite:
		cached = xr.load_dataset(out_path)
		same_times = np.array_equal(cached.time.values, np.array(times, dtype="datetime64[ns]"))
		same_models = list(cached.model.values) == model_names
		required = [f"rmse_{kind}_{var}" for var in kwargs.get("variables", VARIABLES) for kind in ("obs", "anl")]
		if same_times and same_models and all(name in cached for name in required):
			print(f"Loading cached held-out station stats: {out_path}")
			return cached
	print(f"Computing held-out station stats for {len(model_dirs)} models ({len(times)} times)")
	ds = compute_heldout_stats(model_dirs, times, **kwargs)
	save_stats(ds, out_path)
	print(f"Saved {out_path}")
	return ds


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


def plot_heldout_rmse_timeseries(ds, variables, output_dir, show_anl_panel=True):
	"""One PNG per variable: held-out station RMSE vs station obs, plus (optionally) vs RTMA analysis below."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import matplotlib.dates as mdates

	tag = f"{ds.attrs['start_time']}_{ds.attrs['end_time']}"
	n_mean = float(ds.n_heldout.where(ds.n_heldout > 0).mean())
	saved = []
	for var in variables:
		units = UNITS.get(var, "")
		panels = [(f"rmse_obs_{var}", "vs station observations")]
		if show_anl_panel:
			panels.append((f"rmse_anl_{var}", "vs RTMA analysis"))
		fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4 * len(panels)), sharex=True, squeeze=False)
		axes = axes[:, 0]
		for ax, (name, label) in zip(axes, panels):
			# hourly reindex inserts NaNs so missing hours show as gaps
			series = ds[name].resample(time="1h").asfreq()
			for model in series.model.values:
				line = series.sel(model=model)
				ax.plot(line.time.values, line.values, lw=1.2,
						label=f"{model} (mean {float(ds[name].sel(model=model).mean()):.3g})")
			ax.set_title(f"Held-out station RMSE {label}: {var}")
			ax.set_ylabel(f"RMSE [{units}]")
			ax.grid(alpha=0.3)
		ref = ds[f"rmse_rtma_anl_obs_{var}"]
		ref_series = ref.resample(time="1h").asfreq()
		axes[0].plot(ref_series.time.values, ref_series.values, "k--", lw=1.2,
					 label=f"RTMA analysis (mean {float(ref.mean()):.3g})")
		for ax in axes:
			ax.legend(fontsize=8)
		locator = mdates.AutoDateLocator()
		axes[-1].xaxis.set_major_locator(locator)
		axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
		axes[-1].set_xlabel("Valid time (UTC)")
		fig.suptitle(f"Stations held out in: {ds.attrs['holdout_models']} (mean {n_mean:.0f} stations/hour)",
					 fontsize=9)
		fig.tight_layout()
		path = os.path.join(output_dir, f"heldout_rmse_timeseries_{var}_{tag}.png")
		fig.savefig(path, dpi=150)
		plt.close(fig)
		saved.append(path)
	return saved
