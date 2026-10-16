"""Statistical evaluation of ADAF inference output (one NetCDF file per hour, named YYYY-MM-DD_HH.nc)."""

import argparse
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
TOPO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
						 "data_preparation_ges", "RTMA_TOPO_2p5km.nc")
VARIABLES = ("t", "q", "u10", "v10")
UNITS = {"t": "K", "q": "kg kg-1", "u10": "m s-1", "v10": "m s-1"}
TIME_FORMAT = "%Y-%m-%d_%H"
HOLDOUT_SUFFIX = "_hold_out_obs"
GRID_SPACING_KM = 2.5
MAX_RADIUS_KM = 100.0
EARTH_RADIUS_KM = 6371.0


def parse_hour(value):
	try:
		return datetime.strptime(value, TIME_FORMAT)
	except ValueError as exc:
		raise argparse.ArgumentTypeError(f"Invalid hour {value!r}; expected YYYY-MM-DD_HH.") from exc


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


def _radial_offsets(max_radius_km):
	"""Pixel offsets within max_radius_km and their ring index (distance rounded to whole grid cells)."""
	r_max = int(max_radius_km // GRID_SPACING_KM)
	dy, dx = np.mgrid[-r_max:r_max + 1, -r_max:r_max + 1]
	dist = np.hypot(dy, dx)
	keep = dist <= r_max
	return dy[keep], dx[keep], np.rint(dist[keep]).astype(np.intp), r_max + 1


def _radial_sums(values, bins, n_bins):
	"""[sum, sum of squares] per ring."""
	values = values.astype(np.float64)
	return np.stack((np.bincount(bins, values, n_bins), np.bincount(bins, values * values, n_bins)))


def _heldout_chunk(times, model_dirs, holdout_dirs, variables, max_radius_km=MAX_RADIUS_KM):
	"""Held-out station RMSE at the analysis hour, plus errors binned by distance from each station."""
	n_t, n_m, n_v = len(times), len(model_dirs), len(variables)
	rmse_obs = np.full((n_t, n_m, n_v), np.nan)
	rmse_anl = np.full((n_t, n_m, n_v), np.nan)
	rmse_ref = np.full((n_t, n_v), np.nan)
	n_stations = np.zeros(n_t, dtype=np.int32)
	n_excluded = np.zeros(n_t, dtype=np.int32)
	off_y, off_x, off_bin, n_bins = _radial_offsets(max_radius_km)
	rad_count = np.zeros(n_bins)
	rad_obs = np.zeros((n_m, n_v, 2, n_bins))
	rad_anl = np.zeros((n_m, n_v, 2, n_bins))
	rad_ref = np.zeros((n_v, 2, n_bins))
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

		height, width = common.shape
		st_y, st_x = np.unravel_index(idx, common.shape)
		pix_y = st_y[:, None] + off_y[None, :]
		pix_x = st_x[:, None] + off_x[None, :]
		inside = (pix_y >= 0) & (pix_y < height) & (pix_x >= 0) & (pix_x < width)
		pix = (pix_y * width + pix_x)[inside]
		bins = np.broadcast_to(off_bin, inside.shape)[inside]
		station = np.broadcast_to(np.arange(idx.size)[:, None], inside.shape)[inside]
		rad_count += np.bincount(bins, minlength=n_bins)

		for m_idx, model_dir in enumerate(model_dirs):
			path = model_file_path(model_dir, file_time)
			with h5py.File(path, "r") as h5:
				for v_idx, var in enumerate(variables):
					for name in (f"output_{var}", f"output_error_{var}", f"obs_{var}", f"rtma_anl_{var}"):
						if name not in h5:
							raise KeyError(f"{path} is missing {name}.")
					obs = h5[f"obs_{var}"][-1].ravel()[idx].astype(np.float64)
					if not np.isfinite(obs).all():
						raise ValueError(
							f"{path}: obs_{var} is missing at {np.count_nonzero(~np.isfinite(obs))} "
							"held-out station(s); were the models run with different --exclude_metar settings?"
						)
					output = h5[f"output_{var}"][...].ravel()
					error = h5[f"output_error_{var}"][...].ravel()
					rmse_obs[t_idx, m_idx, v_idx] = _rmse(output[idx], obs)
					rmse_anl[t_idx, m_idx, v_idx] = np.sqrt(np.mean(np.square(error[idx].astype(np.float64))))
					diff_obs = output[pix] - obs[station]
					diff_anl = error[pix]
					if not (np.isfinite(diff_obs).all() and np.isfinite(diff_anl).all()):
						raise ValueError(f"{path}: non-finite output_{var} within {max_radius_km} km of a held-out station.")
					rad_obs[m_idx, v_idx] += _radial_sums(diff_obs, bins, n_bins)
					rad_anl[m_idx, v_idx] += _radial_sums(diff_anl, bins, n_bins)
					if m_idx == 0:
						rtma = h5[f"rtma_anl_{var}"][...].ravel()
						rmse_ref[t_idx, v_idx] = _rmse(rtma[idx], obs)
						diff_ref = rtma[pix] - obs[station]
						if not np.isfinite(diff_ref).all():
							raise ValueError(f"{path}: non-finite rtma_anl_{var} within {max_radius_km} km of a held-out station.")
						rad_ref[v_idx] += _radial_sums(diff_ref, bins, n_bins)
	return rmse_obs, rmse_anl, rmse_ref, n_stations, n_excluded, rad_count, rad_obs, rad_anl, rad_ref


def compute_heldout_stats(model_dirs, times, variables=VARIABLES, num_workers=1, max_radius_km=MAX_RADIUS_KM):
	"""RMSE at held-out stations for all models, vs station obs and vs RTMA analysis, as time series
	and as a function of distance from the station (up to max_radius_km).

	Stations are those held out in every *_hold_out_obs model at that hour, so all models are
	scored on the same locations and none of the holdout models saw them.
	"""
	holdout_dirs = [d for d in model_dirs if os.path.normpath(d).endswith(HOLDOUT_SUFFIX)]
	if not holdout_dirs:
		raise ValueError(f"No model directory ends with {HOLDOUT_SUFFIX}.")
	times_per_task = max(1, -(-len(times) // (4 * max(num_workers, 1))))
	chunks = [times[i:i + times_per_task] for i in range(0, len(times), times_per_task)]
	args = [(chunk, model_dirs, holdout_dirs, variables, max_radius_km) for chunk in chunks]
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
	rad_count, rad_obs, rad_anl, rad_ref = (sum(part[i] for part in parts) for i in range(5, 9))
	radius = np.arange(rad_count.size) * GRID_SPACING_KM
	denom = np.where(rad_count > 0, rad_count, np.nan)

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
		"radial_n": xr.DataArray(rad_count.astype(np.int64), dims=("radius",), attrs={
			"long_name": "Number of (station, pixel, hour) samples in each distance ring"}),
	}
	radial_kinds = {
		"obs": (rad_obs, "output_{var} - held-out station obs_{var}"),
		"anl": (rad_anl, "output_{var} - rtma_anl_{var}"),
		"rtma_anl_obs": (rad_ref, "rtma_anl_{var} - held-out station obs_{var}"),
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
		for kind, (sums, description) in radial_kinds.items():
			sums = sums[v_idx] if sums.ndim == 3 else sums[:, v_idx]
			dims = ("radius",) if sums.ndim == 2 else ("model", "radius")
			mean = sums[..., 0, :] / denom
			rmse = np.sqrt(np.maximum(sums[..., 1, :] / denom, 0))
			label = description.format(var=var)
			data_vars[f"radial_rmse_{kind}_{var}"] = xr.DataArray(
				rmse.astype(np.float32), dims=dims,
				attrs={"long_name": f"RMSE of {label} vs distance from held-out station", "units": units},
			)
			data_vars[f"radial_bias_{kind}_{var}"] = xr.DataArray(
				mean.astype(np.float32), dims=dims,
				attrs={"long_name": f"Mean of {label} vs distance from held-out station", "units": units},
			)
	return xr.Dataset(
		data_vars,
		coords={
			"model": np.array(model_names, dtype=object),
			"time": np.array(times, dtype="datetime64[ns]"),
			"radius": ("radius", radius, {
				"units": "km",
				"long_name": f"Distance from held-out station, rounded to the nearest {GRID_SPACING_KM} km grid cell",
			}),
		},
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
		required = [f"{prefix}_{kind}_{var}" for var in kwargs.get("variables", VARIABLES)
					for kind in ("obs", "anl") for prefix in ("rmse", "radial_rmse")]
		same_radius = "radius" in cached.coords and float(cached.radius.max()) == float(
			int(kwargs.get("max_radius_km", MAX_RADIUS_KM) // GRID_SPACING_KM) * GRID_SPACING_KM)
		if same_times and same_models and same_radius and all(name in cached for name in required):
			print(f"Loading cached held-out station stats: {out_path}")
			return cached
	print(f"Computing held-out station stats for {len(model_dirs)} models ({len(times)} times)")
	ds = compute_heldout_stats(model_dirs, times, **kwargs)
	save_stats(ds, out_path)
	print(f"Saved {out_path}")
	return ds


def haversine_km(lat1, lon1, lat2, lon2):
	lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2))
	a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
	return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_stations(path, variables=VARIABLES):
	"""All stations reporting at the analysis hour (finite obs_<var> for any variable).

	Returns a dict of 1-D arrays y, x, lat, lon (degrees east, -180..180) and held_out (None if the
	file has no held_out_obs).
	"""
	with h5py.File(path, "r") as h5:
		for name in ("lat", "lon", *(f"obs_{var}" for var in variables)):
			if name not in h5:
				raise KeyError(f"{path} is missing {name}.")
		present = np.zeros(h5["lat"].shape, dtype=bool)
		for var in variables:
			present |= np.isfinite(h5[f"obs_{var}"][-1])
		y, x = np.nonzero(present)
		lat = h5["lat"][...][y, x].astype(np.float64)
		lon = (h5["lon"][...][y, x].astype(np.float64) + 180) % 360 - 180
		held = h5["held_out_obs"][-1][y, x].astype(bool) if "held_out_obs" in h5 else None
	return {"y": y, "x": x, "lat": lat, "lon": lon, "held_out": held}


def find_stations_near(stations, lat, lon, radius_km):
	"""Stations from load_stations within radius_km of (lat, lon) as a list of dicts, nearest first."""
	dist = haversine_km(lat, lon, stations["lat"], stations["lon"])
	keep = np.flatnonzero(dist <= radius_km)
	keep = keep[np.argsort(dist[keep], kind="stable")]
	nearby = []
	for k in keep:
		station = {
			"y": int(stations["y"][k]), "x": int(stations["x"][k]),
			"lat": float(stations["lat"][k]), "lon": float(stations["lon"][k]),
			"distance_km": float(dist[k]),
		}
		if stations["held_out"] is not None:
			station["held_out"] = bool(stations["held_out"][k])
		nearby.append(station)
	return nearby


def _geodesic_circle(lat, lon, radius_km, n_points=361):
	"""Lat/lon of points radius_km from (lat, lon) on a sphere."""
	lat1, lon1 = np.radians(lat), np.radians(lon)
	bearing = np.linspace(0, 2 * np.pi, n_points)
	d = radius_km / EARTH_RADIUS_KM
	lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(bearing))
	lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
	return np.degrees(lat2), np.degrees(lon2)


def load_topo_window(y, x, half_cells, topo_path=TOPO_PATH):
	"""lat, lon, orog [m] of the RTMA grid window of +-half_cells around grid point (y, x)."""
	with xr.open_dataset(topo_path) as ds:
		window = ds.isel(y=slice(max(y - half_cells, 0), y + half_cells + 1),
						 x=slice(max(x - half_cells, 0), x + half_cells + 1))
		return (window.latitude.values, (window.longitude.values + 180) % 360 - 180, window.orog.values)


def _add_map_borders(ax, scale):
	import cartopy.feature as cfeature
	ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", scale), facecolor="#d4e6f1",
				   edgecolor="#8fb3c9", linewidth=0.4, zorder=0.5)
	ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", scale), facecolor="none",
				   edgecolor="#2b434c", linewidth=0.8, zorder=1)
	ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", scale),
				   facecolor="none", edgecolor="#2b434c", linewidth=0.8, zorder=1)
	ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lakes", scale),
				   facecolor="none", edgecolor="#a7b2ae", linewidth=0.5, zorder=1)
	ax.set_facecolor("#eaf1ed")


def _draw_overview(overview, stations, held, selected, detail_crs, half_m, colors):
	"""CONUS overview: all stations, detail-view box, starred selected station."""
	import cartopy.crs as ccrs
	pc = ccrs.PlateCarree()
	overview.set_extent((-121, -72, 22, 51), crs=pc)
	_add_map_borders(overview, "50m")
	overview.scatter(stations["lon"][~held], stations["lat"][~held], s=1.5, c=colors["visible"],
					 transform=pc, zorder=2, alpha=0.6)
	overview.scatter(stations["lon"][held], stations["lat"][held], s=3, c=colors["held_out"],
					 transform=pc, zorder=2, alpha=0.8)
	box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * half_m
	overview.plot(box[:, 0], box[:, 1], color="#263238", lw=1.4, transform=detail_crs, zorder=4)
	overview.scatter(selected["lon"], selected["lat"], s=160, marker="*", facecolor=colors["star"],
					 edgecolor="#263238", linewidth=0.8, transform=pc, zorder=5)
	overview.set_title("CONUS overview (box = detail view)", loc="left", fontsize=11)


def plot_station_map(stations, selected, search_lat, search_lon, radius_km, out_path, title, subtitle,
					 selected_label="Selected station", max_labels=60, topo_path=TOPO_PATH):
	"""Two-panel station map: large detail view (radius_km + 20% around the selected station) and a CONUS overview.

	The detail view is shaded with RTMA terrain height from topo_path, which must be on the inference grid.
	Held-out stations are drawn separately when stations["held_out"] is available; stations in the detail
	view are labelled with (y, x) when there are at most max_labels of them.
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap
	from matplotlib.lines import Line2D
	import cartopy.crs as ccrs

	colors = {"visible": "#177b74", "held_out": "#c83f40", "star": "#f7e27a", "search": "#3a4f9c"}
	held = stations["held_out"] if stations["held_out"] is not None else np.zeros(stations["y"].size, bool)
	half_m = 1.2 * radius_km * 1000.0
	detail_crs = ccrs.AzimuthalEquidistant(central_longitude=selected["lon"], central_latitude=selected["lat"])
	pc = ccrs.PlateCarree()

	fig = plt.figure(figsize=(13.5, 8), facecolor="#f6f8f6")
	# detail width matches its height so the square map fills the axes
	detail_box = (0.045, 0.11, 0.72 * 8 / 13.5, 0.72)
	detail = fig.add_axes(detail_box, projection=detail_crs)
	cbar_ax = fig.add_axes((detail_box[0] + detail_box[2] + 0.008, detail_box[1], 0.012, detail_box[3]))
	overview = fig.add_axes((0.55, 0.30, 0.44, 0.50), projection=ccrs.LambertConformal(
		central_longitude=-96, central_latitude=39, standard_parallels=(33, 45)))
	overview.set_anchor("W")

	# detail panel
	detail.set_extent((-half_m, half_m, -half_m, half_m), crs=detail_crs)
	_add_map_borders(detail, "10m")
	half_cells = int(np.ceil(half_m * 1.5e-3 / GRID_SPACING_KM)) + 2
	topo_lat, topo_lon, orog = load_topo_window(selected["y"], selected["x"], half_cells, topo_path)
	centre = (selected["y"] - max(selected["y"] - half_cells, 0), selected["x"] - max(selected["x"] - half_cells, 0))
	if abs(topo_lat[centre] - selected["lat"]) > 1e-3 or abs(topo_lon[centre] - selected["lon"]) > 1e-3:
		raise ValueError(f"{topo_path} is not on the inference grid.")
	# drop the blue (sea) end of the terrain colormap and pre-blend with white (alpha on a mesh shows cell seams)
	land = plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256))
	land[:, :3] = 0.75 * land[:, :3] + 0.25
	terrain = LinearSegmentedColormap.from_list("land", land)
	mesh = detail.pcolormesh(topo_lon, topo_lat, orog, cmap=terrain, shading="nearest",
							 linewidth=0, antialiased=False, transform=pc, zorder=0, rasterized=True)
	cbar = fig.colorbar(mesh, cax=cbar_ax)
	cbar.set_label("Terrain height [m]", fontsize=10)
	cbar.ax.tick_params(labelsize=9)
	in_view = haversine_km(selected["lat"], selected["lon"], stations["lat"], stations["lon"]) <= half_m * 1.5e-3
	is_selected = (stations["y"] == selected["y"]) & (stations["x"] == selected["x"])
	for label, mask, marker, size in (("visible", ~held, "o", 28), ("held_out", held, "X", 55)):
		pick = in_view & mask & ~is_selected
		detail.scatter(stations["lon"][pick], stations["lat"][pick], s=size, c=colors[label], marker=marker,
					   edgecolor="white", linewidths=0.4, transform=pc, zorder=3)
	if np.count_nonzero(in_view) <= max_labels:
		for k in np.flatnonzero(in_view & ~is_selected):
			detail.annotate(f"({stations['y'][k]}, {stations['x'][k]})",
							detail_crs.transform_point(stations["lon"][k], stations["lat"][k], pc),
							xytext=(4, 4), textcoords="offset points", fontsize=7, color="#23383a", zorder=4,
							annotation_clip=True)
	circle_lat, circle_lon = _geodesic_circle(search_lat, search_lon, radius_km)
	detail.plot(circle_lon, circle_lat, "--", color=colors["search"], lw=1.4, transform=pc, zorder=2)
	detail.scatter(search_lon, search_lat, s=90, marker="+", c=colors["search"], linewidths=1.6,
				   transform=pc, zorder=4)
	detail.scatter(selected["lon"], selected["lat"], s=320, marker="*", facecolor=colors["star"],
				   edgecolor="#263238", linewidth=1.0, transform=pc, zorder=5)
	grid = detail.gridlines(draw_labels=True, linewidth=0.3, color="#90a4ae", alpha=0.7,
							x_inline=False, y_inline=False)
	grid.top_labels = grid.right_labels = False
	grid.xlabel_style = grid.ylabel_style = {"size": 9}

	# overview panel
	_draw_overview(overview, stations, held, selected, detail_crs, half_m, colors)

	fig.suptitle(title, x=0.02, y=0.97, ha="left", fontsize=18, fontweight="bold", color="#23383a")
	fig.text(0.02, 0.905, subtitle, fontsize=10.5, color="#405459", va="top")
	visible_label = "Assimilated station" if stations["held_out"] is not None else "Station"
	legend = [Line2D([0], [0], marker="o", linestyle="none", markersize=7, color=colors["visible"],
					 label=f"{visible_label} ({np.count_nonzero(~held):,})")]
	if stations["held_out"] is not None:
		legend.append(Line2D([0], [0], marker="X", linestyle="none", markersize=8, color=colors["held_out"],
							 label=f"Held-out station ({np.count_nonzero(held):,})"))
	legend += [
		Line2D([0], [0], marker="*", linestyle="none", markersize=15, markerfacecolor=colors["star"],
			   markeredgecolor="#263238", label=f"{selected_label} ({selected['y']}, {selected['x']})"),
		Line2D([0], [0], marker="+", color=colors["search"], linestyle="--", markersize=10,
			   label=f"Search point and {radius_km:g} km radius"),
	]
	fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.56, 0.12), frameon=False, fontsize=10)
	os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
	fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
	plt.close(fig)
	return out_path


def plot_denial_map(diff, lat, lon, stations, selected, half_width_km, out_path, title, subtitle,
					colorbar_label, topo_path=TOPO_PATH):
	"""Station-map layout showing a full-domain difference field (e.g. with minus without a station).

	Detail view: diff within half_width_km of the selected (denied) station on a symmetric diverging scale,
	terrain contours from topo_path and reporting stations; CONUS overview with the detail box.
	lat/lon are the full-domain grid (lon in -180..180).
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	from matplotlib.ticker import MaxNLocator
	import cartopy.crs as ccrs

	colors = {"visible": "#263238", "held_out": "#c83f40", "star": "#f7e27a"}
	half_m = half_width_km * 1000.0
	detail_crs = ccrs.AzimuthalEquidistant(central_longitude=selected["lon"], central_latitude=selected["lat"])
	pc = ccrs.PlateCarree()

	fig = plt.figure(figsize=(13.5, 8), facecolor="#f6f8f6")
	detail_box = (0.045, 0.11, 0.72 * 8 / 13.5, 0.72)
	detail = fig.add_axes(detail_box, projection=detail_crs)
	cbar_ax = fig.add_axes((detail_box[0] + detail_box[2] + 0.008, detail_box[1], 0.012, detail_box[3]))
	overview = fig.add_axes((0.55, 0.30, 0.44, 0.50), projection=ccrs.LambertConformal(
		central_longitude=-96, central_latitude=39, standard_parallels=(33, 45)))
	overview.set_anchor("W")

	detail.set_extent((-half_m, half_m, -half_m, half_m), crs=detail_crs)
	_add_map_borders(detail, "10m")
	# 1.5x covers the box corners (sqrt 2) plus grid rotation
	half_cells = int(np.ceil(half_m * 1.5e-3 / GRID_SPACING_KM)) + 2
	window = (slice(max(selected["y"] - half_cells, 0), selected["y"] + half_cells + 1),
			  slice(max(selected["x"] - half_cells, 0), selected["x"] + half_cells + 1))
	win_diff, win_lat, win_lon = diff[window], lat[window], lon[window]
	in_box = haversine_km(selected["lat"], selected["lon"], win_lat, win_lon) <= half_width_km * np.sqrt(2)
	vmax = float(np.nanmax(np.abs(win_diff[in_box]))) or 1e-12
	mesh = detail.pcolormesh(win_lon, win_lat, win_diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest",
							 linewidth=0, antialiased=False, transform=pc, zorder=0.6, rasterized=True)
	cbar = fig.colorbar(mesh, cax=cbar_ax)
	cbar.set_label(colorbar_label, fontsize=10)
	cbar.ax.tick_params(labelsize=9)

	topo_lat, topo_lon, orog = load_topo_window(selected["y"], selected["x"], half_cells, topo_path)
	levels = MaxNLocator(8).tick_values(float(np.nanmin(orog)), float(np.nanmax(orog)))
	detail.contour(topo_lon, topo_lat, orog, levels=levels, colors="#5d4037", linewidths=0.5, alpha=0.6,
				   transform=pc, zorder=1.5)

	in_view = haversine_km(selected["lat"], selected["lon"], stations["lat"], stations["lon"]) <= half_m * 1.5e-3
	is_selected = (stations["y"] == selected["y"]) & (stations["x"] == selected["x"])
	pick = in_view & ~is_selected
	detail.scatter(stations["lon"][pick], stations["lat"][pick], s=14, c=colors["visible"], marker="o",
				   edgecolor="white", linewidths=0.4, transform=pc, zorder=3)
	detail.scatter(selected["lon"], selected["lat"], s=320, marker="*", facecolor=colors["star"],
				   edgecolor="#263238", linewidth=1.0, transform=pc, zorder=5)
	grid = detail.gridlines(draw_labels=True, linewidth=0.3, color="#90a4ae", alpha=0.7,
							x_inline=False, y_inline=False)
	grid.top_labels = grid.right_labels = False
	grid.xlabel_style = grid.ylabel_style = {"size": 9}

	_draw_overview(overview, stations, np.zeros(stations["y"].size, bool), selected, detail_crs, half_m, colors)

	fig.suptitle(title, x=0.02, y=0.97, ha="left", fontsize=18, fontweight="bold", color="#23383a")
	fig.text(0.02, 0.905, subtitle, fontsize=10.5, color="#405459", va="top")
	legend = [
		Line2D([0], [0], marker="o", linestyle="none", markersize=5, color=colors["visible"],
			   label=f"Station reporting at analysis hour ({stations['y'].size:,})"),
		Line2D([0], [0], marker="*", linestyle="none", markersize=15, markerfacecolor=colors["star"],
			   markeredgecolor="#263238", label=f"Denied station ({selected['y']}, {selected['x']})"),
		Line2D([0], [0], color="#5d4037", lw=0.8, alpha=0.6,
			   label=f"Terrain height every {levels[1] - levels[0]:g} m"),
	]
	fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.56, 0.12), frameon=False, fontsize=10)
	os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
	fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
	plt.close(fig)
	return out_path


def slice_indices(lat, lon, y, x, along, half_width_km):
	"""Grid points nearest a line of constant latitude (along="lat") or longitude (along="lon")
	through grid point (y, x), within half_width_km of it.

	For a latitude slice each grid column contributes the row whose lat is closest to lat[y, x]
	(and vice versa for longitude), so the slice passes exactly through (y, x). Returns (ys, xs)
	ordered by increasing longitude (lat slice) or latitude (lon slice).
	"""
	height, width = lat.shape
	half_cells = int(np.ceil(half_width_km * 1.5 / GRID_SPACING_KM)) + 2
	rows = np.arange(max(y - half_cells, 0), min(y + half_cells + 1, height))
	cols = np.arange(max(x - half_cells, 0), min(x + half_cells + 1, width))
	if along == "lat":
		xs = cols
		ys = rows[np.argmin(np.abs(lat[np.ix_(rows, cols)] - lat[y, x]), axis=0)]
		order_key = lon[ys, xs]
	elif along == "lon":
		ys = rows
		dlon = (lon[np.ix_(rows, cols)] - lon[y, x] + 180) % 360 - 180
		xs = cols[np.argmin(np.abs(dlon), axis=1)]
		order_key = lat[ys, xs]
	else:
		raise ValueError(f"along must be 'lat' or 'lon', got {along!r}.")
	keep = haversine_km(lat[y, x], lon[y, x], lat[ys, xs], lon[ys, xs]) <= half_width_km
	ys, xs, order_key = ys[keep], xs[keep], order_key[keep]
	order = np.argsort(order_key)
	return ys[order], xs[order]


def plot_denial_slices(with_station, without_station, lat, lon, selected, obs, half_width_km, out_paths,
					   title, subtitle, units, stations=None):
	"""Latitude and longitude slices through the denied station (out_paths = {"lat": path, "lon": path}).

	Top panel: field with and without the station along the slice (plus the station's obs);
	bottom panel: with minus without. Stations from load_stations lying on slice pixels are marked.
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	y, x = selected["y"], selected["x"]
	station_pixels = set() if stations is None else set(zip(stations["y"].tolist(), stations["x"].tolist()))
	saved = {}
	for along, out_path in out_paths.items():
		ys, xs = slice_indices(lat, lon, y, x, along, half_width_km)
		if along == "lat":
			coord, centre, coord_label = lon[ys, xs], selected["lon"], "Longitude [deg E]"
			km_per_deg = np.pi / 180 * EARTH_RADIUS_KM * np.cos(np.radians(selected["lat"]))
			slice_label = f"latitude {selected['lat']:.4f}"
			dir_label = "east"
		else:
			coord, centre, coord_label = lat[ys, xs], selected["lat"], "Latitude [deg N]"
			km_per_deg = np.pi / 180 * EARTH_RADIUS_KM
			slice_label = f"longitude {selected['lon']:.4f}"
			dir_label = "north"
		on_slice_stations = np.array([(yy, xx) in station_pixels and (yy, xx) != (y, x) for yy, xx in zip(ys, xs)])
		with_vals, without_vals = with_station[ys, xs], without_station[ys, xs]
		diff = with_vals - without_vals

		fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
										  gridspec_kw={"height_ratios": (3, 2)})
		top.plot(coord, with_vals, color="#1f5f99", lw=1.6, label="With station")
		top.plot(coord, without_vals, color="#d9822b", lw=1.6, ls="--", label="Station denied")
		if np.isfinite(obs):
			top.scatter(centre, obs, s=150, marker="*", facecolor="#f7e27a", edgecolor="#263238", zorder=5,
						label=f"Station obs ({obs:.4g})")
		bottom.plot(coord, diff, color="#263238", lw=1.6)
		bottom.axhline(0, color="#90a4ae", lw=0.8)
		for ax in (top, bottom):
			ax.axvline(centre, color="#c83f40", lw=0.9, ls=":", zorder=0)
			if on_slice_stations.any():
				ax.scatter(coord[on_slice_stations], (with_vals if ax is top else diff)[on_slice_stations],
						   s=22, facecolor="white", edgecolor="#263238", zorder=4,
						   label="Other station on slice" if ax is top else None)
			ax.grid(alpha=0.3)
		top.set_ylabel(f"Analysis [{units}]")
		top.legend(fontsize=9, loc="best")
		top.set_title(f"Slice along {slice_label} (nearest grid point per "
					  f"{'column' if along == 'lat' else 'row'}), +-{half_width_km:g} km", fontsize=10)
		bottom.set_ylabel(f"With minus without [{units}]")
		bottom.set_xlabel(coord_label)
		secondary = top.secondary_xaxis("top", functions=(lambda c, c0=centre, k=km_per_deg: (c - c0) * k,
														   lambda d, c0=centre, k=km_per_deg: c0 + d / k))
		secondary.set_xlabel(f"Distance {dir_label} of station [km]", fontsize=9)
		fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color="#23383a")
		fig.text(0.01, 0.93, subtitle, fontsize=9, color="#405459", va="top")
		fig.tight_layout(rect=(0, 0, 1, 0.925))
		os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
		fig.savefig(out_path, dpi=150)
		plt.close(fig)
		saved[along] = out_path
	return saved


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


def plot_heldout_radial(ds, variables, output_dir):
	"""Per variable, two PNGs of RMSE vs distance from held-out stations:
	heldout_radial_obs_* (models and RTMA analysis vs station obs) and heldout_radial_anl_* (models vs RTMA analysis).
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	tag = f"{ds.attrs['start_time']}_{ds.attrs['end_time']}"
	subtitle = (f"{int(ds.n_heldout.sum())} held-out station-hours, "
				f"{len(ds.attrs['holdout_models'].split(', '))} holdout model(s)")
	saved = []
	for var in variables:
		units = UNITS.get(var, "")
		for kind, title in (("obs", "difference from held-out station observation"),
							("anl", "difference from RTMA analysis")):
			fig, ax = plt.subplots(figsize=(9, 5.5))
			field = ds[f"radial_rmse_{kind}_{var}"]
			for model in field.model.values:
				ax.plot(ds.radius.values, field.sel(model=model).values, lw=1.4, marker=".", ms=4, label=model)
			if kind == "obs":
				ax.plot(ds.radius.values, ds[f"radial_rmse_rtma_anl_obs_{var}"].values, "k--", lw=1.4,
						label="RTMA analysis")
			ax.set_title(f"RMSE of {title} vs distance: {var}\n{subtitle}", fontsize=10)
			ax.set_xlabel("Distance from held-out station [km]")
			ax.set_ylabel(f"RMSE [{units}]")
			ax.set_xlim(0, float(ds.radius.max()))
			ax.grid(alpha=0.3)
			ax.legend(fontsize=8)
			fig.tight_layout()
			path = os.path.join(output_dir, f"heldout_radial_{kind}_{var}_{tag}.png")
			fig.savefig(path, dpi=150)
			plt.close(fig)
			saved.append(path)
	return saved
