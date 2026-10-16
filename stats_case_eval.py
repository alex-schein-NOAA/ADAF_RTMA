r"""Single-case (single time / single station) evaluation of ADAF inference output.

Example (run from ADAF_RTMA with ADAF_environment active; all plots go to ADAF_RTMA/stats_results/):

	python stats_case_eval.py --station_planning \
		--model lowres_21563899_resume_22722533_hold_out_obs \
		--time 2023-01-05_12 --lat 39.74 --lon -104.99 --radius_km 50

With --station_planning (and nothing else is run), lists every station reporting at the analysis hour
within --radius_km of the search point: its grid
(y, x) location, lat/lon, distance, and (for _hold_out_obs models) whether it was held out.
The search point is (--lat, --lon), or the --station location if lat/lon are omitted.
The selected station is --station Y X if given, otherwise the station closest to the search point; it is
starred on a map saved to station_map_tmp.png (detail view of radius + 20% plus a CONUS overview).

Without --station_planning, runs single-station data denial for --station Y X (lat/lon are not used):
inference is run twice on the model's checkpoint (read from the inference output file) with random holdout
off, (a) with all stations and (b) with the station removed from the model input for --deny_hours
(0 = analysis hour, 1 and 2 = earlier hours; default all). For each variable, (a) - (b) is mapped within
--radius_km (default 100 km, map half-width only) of the station and saved to denial_map_<var>_tmp.png.
Latitude/longitude slices through the station (nearest grid point to the line, +-radius_km) are saved to
denial_lat_slice_<var>.png and denial_lon_slice_<var>.png: field with and without the station (top) and
their difference (bottom).
Needs a GPU in practice:

	python stats_case_eval.py --model lowres_21563899_resume_22722533 --time 2023-01-05_12 \
		--station 768 948 --deny_hours 0

Denial map header (also printed as a table), all values at the analysis hour:
  obs              the station's observation
  with             analysis (a), all stations assimilated, at the station's grid point
  without          analysis (b), station denied, at the station's grid point
  max |diff|       largest |with - without| anywhere in the domain, with its distance from the station;
                   a large distance means the denial changed the analysis far from the station
q is in g/kg (run_model_inference scales it), unlike the kg/kg of the stored inference files.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.YParams import YParams
from utils.inference_functions import load_checkpoint_weights, run_model_inference
from utils.stats_functions import (
	DEFAULT_DATA_ROOT,
	UNITS,
	VARIABLES,
	find_stations_near,
	haversine_km,
	load_stations,
	model_file_path,
	parse_hour,
	plot_denial_map,
	plot_denial_slices,
	plot_station_map,
)

PROJECT_DIR = Path(__file__).parent
PLOT_DIR = PROJECT_DIR / "stats_results"
STATION_MAP_PATH = PLOT_DIR / "station_map_tmp.png"
DENIAL_MAP_PATH = str(PLOT_DIR / "denial_map_{var}_tmp.png")
DENIAL_SLICE_PATH = str(PLOT_DIR / "denial_{along}_slice_{var}.png")
DEFAULT_INPUT_DIR = "/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes/test_data"
DEFAULT_DENIAL_RADIUS_KM = 100.0
# run_model_inference returns q in g/kg
CASE_UNITS = {**UNITS, "q": "g kg-1"}
CASE_SCALE = {"q": 1000.0}


def get_args():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--model", required=True, help="Model subfolder name under --data_root.")
	parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
	parser.add_argument("--time", type=parse_hour, required=True, help="Analysis hour, YYYY-MM-DD_HH.")
	parser.add_argument("--station_planning", action="store_true",
						help="Only list nearby stations and plot the station map, then exit.")
	parser.add_argument("--lat", type=float, help="Search point, degrees north.")
	parser.add_argument("--lon", type=float, help="Search point, degrees east; -180..180 or 0..360.")
	parser.add_argument("--station", type=int, nargs=2, metavar=("Y", "X"),
						help="Grid (y, x) of the station; in planning mode defaults to the station nearest the search point.")
	parser.add_argument("--radius_km", type=float,
						help="Planning: search radius (required). Denial: half-width of the map "
							 f"(default {DEFAULT_DENIAL_RADIUS_KM:g}).")
	parser.add_argument("--deny_hours", type=int, nargs="+", default=None,
						help="Hours to deny the station: 0 = analysis hour, 1, 2 = earlier (default: all).")
	parser.add_argument("--variables", nargs="+", default=list(VARIABLES), choices=VARIABLES)
	parser.add_argument("--config_filepath", default=str(PROJECT_DIR / "config/params_lowres_ges_goes.yaml"))
	parser.add_argument("--checkpoint_path", default=None,
						help="Default: the checkpoint recorded in the model's inference output file.")
	parser.add_argument("--stats_path", default=str(PROJECT_DIR / "data_preparation_ges/stats_ges.csv"))
	parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory of model input files.")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()
	if not args.station_planning:
		if args.station is None:
			parser.error("--station Y X is required unless --station_planning is given.")
		if args.radius_km is None:
			args.radius_km = DEFAULT_DENIAL_RADIUS_KM
		if args.radius_km <= 0:
			parser.error("--radius_km must be positive.")
		return args
	if args.radius_km is None:
		parser.error("--radius_km is required with --station_planning.")
	if (args.lat is None) != (args.lon is None):
		parser.error("--lat and --lon must be given together.")
	if args.lat is None and args.station is None:
		parser.error("Give --lat/--lon, --station, or both.")
	if args.lat is not None and not -90 <= args.lat <= 90:
		parser.error("--lat must be within -90..90.")
	if args.radius_km <= 0:
		parser.error("--radius_km must be positive.")
	return args


def print_stations(stations, selected):
	has_held = stations and "held_out" in stations[0]
	header = f"  {'y':>5} {'x':>5} {'lat':>9} {'lon':>10} {'dist_km':>8}"
	print(header + (f" {'held_out':>8}" if has_held else ""))
	for s in stations:
		mark = "*" if (s["y"], s["x"]) == (selected["y"], selected["x"]) else " "
		line = f"{mark} {s['y']:>5d} {s['x']:>5d} {s['lat']:>9.4f} {s['lon']:>10.4f} {s['distance_km']:>8.2f}"
		print(line + (f" {'yes' if s['held_out'] else '':>8}" if has_held else ""))


def station_planning(args, stations):
	if args.station is not None:
		match = (stations["y"] == args.station[0]) & (stations["x"] == args.station[1])
		if not match.any():
			raise ValueError(f"No station reports at (y, x) = {tuple(args.station)} at {args.time:%Y-%m-%d_%H}.")
		if args.lat is None:
			search_lat, search_lon = float(stations["lat"][match][0]), float(stations["lon"][match][0])
		else:
			search_lat, search_lon = args.lat, args.lon
		only_match = {key: None if v is None else v[match] for key, v in stations.items()}
		selected = find_stations_near(only_match, search_lat, search_lon, np.inf)[0]
	else:
		search_lat, search_lon = args.lat, args.lon
		selected = find_stations_near(stations, search_lat, search_lon, np.inf)[0]
	search_lon = (search_lon + 180) % 360 - 180
	aligned = (abs(selected["lat"] - search_lat) <= 1e-4
			   and abs((selected["lon"] - search_lon + 180) % 360 - 180) <= 1e-4)
	label = "Selected station" if args.station is not None or aligned else "Closest station"

	nearby = find_stations_near(stations, search_lat, search_lon, args.radius_km)
	print(f"{len(nearby)} station(s) within {args.radius_km:g} km of ({search_lat:.4f}, {search_lon:.4f}) "
		  f"at {args.time:%Y-%m-%d_%H} in {args.model}")
	if nearby:
		print_stations(nearby, selected)
	print(f"{label} (*): (y, x) = ({selected['y']}, {selected['x']}), "
		  f"{selected['lat']:.4f}, {selected['lon']:.4f}, {selected['distance_km']:.2f} km from search point")

	n_held = sum(s.get("held_out", False) for s in nearby)
	title = (f"Station (y, x) = ({selected['y']}, {selected['x']})  |  "
			 f"lat {selected['lat']:.4f}, lon {selected['lon']:.4f}"
			 + ("  (held out)" if selected.get("held_out") else ""))
	subtitle = (f"{args.model}  |  {args.time:%Y-%m-%d_%H}\n"
				f"Search point ({search_lat:.4f}, {search_lon:.4f}), radius {args.radius_km:g} km: {len(nearby)} station(s)"
				+ (f", {n_held} held out" if stations["held_out"] is not None else "")
				+ f"  |  {label.lower()} {selected['distance_km']:.1f} km from search point")
	plot_station_map(stations, selected, search_lat, search_lon, args.radius_km, STATION_MAP_PATH, title, subtitle,
					 selected_label=label)
	print(f"Saved {STATION_MAP_PATH}")


def load_model(params, checkpoint_path, device):
	if params.arch == "lowres":
		from models.encdec_lowres import LowResEncDec as Model
	else:
		from models.encdec import EncDec as Model
	model = Model(params).to(device)
	_, missing, unexpected = load_checkpoint_weights(model, checkpoint_path, device)
	missing_weights = [key for key in missing if "attn_mask" not in key]
	if missing_weights or unexpected:
		raise RuntimeError(f"Checkpoint mismatch: missing={missing_weights}, unexpected={unexpected}")
	return model.eval()


def station_denial(args, path):
	y, x = args.station
	with h5py.File(path, "r") as h5:
		checkpoint = args.checkpoint_path or h5.attrs.get("checkpoint")
		input_name = h5.attrs.get("input_file", os.path.basename(path))
		lat = h5["lat"][...].astype(np.float64)
		lon = (h5["lon"][...].astype(np.float64) + 180) % 360 - 180
		# obs_<var>: (obs_time_window, y, x), last entry = analysis hour
		station_obs = {var: h5[f"obs_{var}"][:, y, x].astype(np.float64) for var in args.variables}
	if not checkpoint:
		raise ValueError(f"{path} records no checkpoint; pass --checkpoint_path.")
	n_window = next(iter(station_obs.values())).size
	hours = sorted(set(range(n_window) if args.deny_hours is None else args.deny_hours))
	if any(h < 0 or h >= n_window for h in hours):
		raise ValueError(f"--deny_hours must be within 0..{n_window - 1}.")
	reported = [h for h in range(n_window) if any(np.isfinite(obs[-(h + 1)]) for obs in station_obs.values())]
	if not set(hours) & set(reported):
		raise ValueError(f"Station ({y}, {x}) has no observations at the denied hour(s) {hours} "
						 f"(it reports at hour(s) {reported}); nothing to deny.")
	input_file = os.path.join(args.input_dir, input_name)
	if not os.path.exists(input_file):
		raise FileNotFoundError(f"Model input file not found: {input_file}")

	params = YParams(args.config_filepath)
	params.hold_out_obs = False
	device = torch.device(args.device)
	if device.type != "cuda":
		print("WARNING: running on CPU; full-domain inference will be very slow.")
	print(f"Checkpoint: {checkpoint}\nInput: {input_file}\nDevice: {device}")
	model = load_model(params, checkpoint, device)
	full = run_model_inference(model, input_file, params, args.stats_path, device)
	denied = run_model_inference(model, input_file, params, args.stats_path, device,
								 deny_station=(y, x), deny_station_hours=hours)
	n_changed = np.count_nonzero(full["input_array_norm"] != denied["input_array_norm"])
	if n_changed == 0:
		raise RuntimeError("Denial did not change the model input.")

	selected = {"y": y, "x": x, "lat": float(lat[y, x]), "lon": float(lon[y, x])}
	stations = load_stations(path)
	stations["held_out"] = None
	dist = haversine_km(selected["lat"], selected["lon"], lat, lon)
	hour_text = "all hours" if len(hours) == n_window else "hour(s) " + ", ".join(map(str, hours))
	print(f"Station ({y}, {x}) at {selected['lat']:.4f}, {selected['lon']:.4f} denied for {hour_text} "
		  f"(reports at hour(s) {reported}; 0 = analysis hour); {n_changed} input values changed.")
	print(f"{'var':>4} {'obs':>10} {'with':>10} {'without':>10} {'diff@stn':>10} {'max|diff|':>10} {'at_km':>8}")

	names = full["output_channel_names"]
	saved = []
	for var in args.variables:
		channel = names.index(f"output_{var}")
		with_station = full["prediction_analysis_unnorm"][channel]
		without_station = denied["prediction_analysis_unnorm"][channel]
		diff = with_station - without_station
		abs_diff = np.abs(diff)
		peak = np.unravel_index(np.argmax(abs_diff), diff.shape)
		obs = station_obs[var][-1] * CASE_SCALE.get(var, 1.0)
		units = CASE_UNITS[var]
		print(f"{var:>4} {obs:>10.4g} {with_station[y, x]:>10.4g} {without_station[y, x]:>10.4g} {diff[y, x]:>10.3g} "
			  f"{abs_diff[peak]:>10.3g} {dist[peak]:>8.1f}")
		title = f"Station denial impact on {var}: ({y}, {x})  |  lat {selected['lat']:.4f}, lon {selected['lon']:.4f}"
		subtitle = (f"{args.model}  |  {args.time:%Y-%m-%d_%H}  |  denied {hour_text} (0 = analysis hour)  |  "
					f"random holdout off\n"
					f"At station: obs {obs:.4g}, with {with_station[y, x]:.4g}, without {without_station[y, x]:.4g} "
					f"{units}  |  max |diff| {abs_diff[peak]:.3g} ({dist[peak]:.1f} km from station)")
		out_path = DENIAL_MAP_PATH.format(var=var)
		plot_denial_map(diff, lat, lon, stations, selected, args.radius_km, out_path, title, subtitle,
						f"With minus without station [{units}]")
		saved.append(out_path)
		slice_paths = {along: DENIAL_SLICE_PATH.format(along=along, var=var) for along in ("lat", "lon")}
		saved += plot_denial_slices(with_station, without_station, lat, lon, selected, obs, args.radius_km,
									slice_paths, title, subtitle, units, stations).values()
	for out_path in saved:
		print(f"Saved {out_path}")


def main():
	args = get_args()
	path = model_file_path(os.path.join(args.data_root, args.model), args.time)
	if not os.path.exists(path):
		raise FileNotFoundError(f"No file for {args.time:%Y-%m-%d_%H}: {path}")
	if args.station_planning:
		station_planning(args, load_stations(path))
		return
	station_denial(args, path)


if __name__ == "__main__":
	main()
