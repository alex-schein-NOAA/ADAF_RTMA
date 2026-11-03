r"""Single-case (single time / single station) evaluation of ADAF inference output.

Example (run from ADAF_RTMA with ADAF_environment active; all plots go under ADAF_RTMA/stats_results/):

	python stats_case_eval.py --station_planning \
		--model lowres_21563899_resume_22722533_hold_out_obs \
		--time 2023-01-05_12 --lat 39.74 --lon -104.99 --radius_km 50

With --station_planning (and nothing else is run), lists every station reporting at the analysis hour
within --radius_km of the search point: its grid
(y, x) location, lat/lon, distance, and (for _hold_out_obs models) whether it was held out.
The search point is (--lat, --lon), or the --station location if lat/lon are omitted.
The selected station is --station Y X if given, otherwise the station closest to the search point; it is
starred on a map saved to stats_results/case_studies/station_map.png (detail view of radius + 20% plus a CONUS overview).

Without --station_planning, runs single-station data denial for --station Y X (lat/lon are not used):
inference is run twice on the model's checkpoint (read from the inference output file) with random holdout
off, (a) with all stations and (b) with the station removed from the model input for --deny_hours
(0 = analysis hour, 1 and 2 = earlier hours; default all). For each variable, (a) - (b) is mapped within
--radius_km (default 100 km, map half-width only) of the station and saved to denial_map_<var>.png.
With --other_stations_denied (default off) every other station is also removed from the model input (all hours)
in both runs, so (a) is the requested station alone and (b) has no stations: the map then shows what this one
station does by itself, not its marginal effect on top of the other stations. The stations drawn on the maps and
slices are still all stations in the file, although the model does not see the others in this mode.
Innovation maps (model minus RTMA analysis, same two-panel layout) for the two runs are saved to
innovation_map_with_station_<var>.png and innovation_map_without_station_<var>.png; their titles give the obs, the
model value, the RTMA analysis value and the innovation at the station, and the largest |innovation| within --radius_km.
Full-field maps of the model output with and without the station, the RTMA first guess and the RTMA analysis are saved
to output_map_model_with_station_<var>.png, output_map_model_without_station_<var>.png, output_map_ges_<var>.png and
output_map_anl_<var>.png (same layout; the four maps of a variable share one colour range, diverging and centred on
zero for u10/v10); turn them off with --no_full_output_map_plots.
Latitude/longitude slices through the station (nearest grid point to the line, +-radius_km) are saved to
denial_lat_slice_<var>.png and denial_lon_slice_<var>.png: field with and without the station (top, together with
the RTMA background and analysis as references) and their difference (bottom). The RTMA lines can be turned off
with --no-slice_plots_include_ges / --no-slice_plots_include_anl (both on by default).
All denial plots keep the file names above and are saved under stats_results/case_studies/ in
	<model>/station_denial/station_x_<x>_y_<y>/<YYYY-MM-DD_HH>/other_stations_<denied|included>/
(other_stations_denied when --other_stations_denied is given, otherwise other_stations_included); the folder is
created as needed and printed at the start of the run. Rerunning the same case overwrites its files.
The console output of a denial run (the command, the denial table, the innovation values and 3x3 blocks, the saved
file list) is also written to stats_output.txt in that folder; station planning does not write one.
Needs a GPU in practice:

	python stats_case_eval.py --model lowres_21563899_resume_22722533 --time 2023-01-05_12 \
		--station 768 948 --deny_hours 0

Denial map header (also printed as a table), all values at the analysis hour:
  obs              the station's observation
  with             analysis (a), all stations assimilated, at the station's grid point (with --other_stations_denied:
                   only this station assimilated)
  without          analysis (b), station denied, at the station's grid point (with --other_stations_denied: no
                   stations assimilated)
  max |diff|       largest |with - without| anywhere in the domain, with its distance from the station;
                   a large distance means the denial changed the analysis far from the station
q is in g/kg (run_model_inference scales it), unlike the kg/kg of the stored inference files.
"""

import argparse
import contextlib
import os
import sys
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
	denial_map_vmax,
	field_map_limits,
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
CASE_DIR = PLOT_DIR / "case_studies"
STATION_MAP_PATH = CASE_DIR / "station_map.png"
# copy of the console output of a denial run, saved with its plots
STATS_OUTPUT_NAME = "stats_output.txt"
# Denial plots go to CASE_DIR/<model>/station_denial/station_x_<x>_y_<y>/<YYYY-MM-DD_HH>/other_stations_<denied|included>/
DENIAL_MAP_NAME = "denial_map_{var}.png"
INNOVATION_MAP_NAME = "innovation_map_{which}_station_{var}.png"
OUTPUT_MAP_NAME = "output_map_{which}_{var}.png"
DENIAL_SLICE_NAME = "denial_{along}_slice_{var}.png"
DEFAULT_INPUT_DIR = "/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes/test_data"
DEFAULT_DENIAL_RADIUS_KM = 100.0
# half-width in grid cells of the innovation block printed around the station (1 -> 3x3)
NEIGHBOURHOOD_HALF_WIDTH = 1
# run_model_inference returns q in g/kg; t is in degrees C (data prep converts K to C)
CASE_UNITS = {**UNITS, "q": "g kg-1", "t": "\u00b0C"}
CASE_SCALE = {"q": 1000.0}
# full-field output maps: wind components are signed (diverging colours centred on zero), t and q are not
SIGNED_VARS = ("u10", "v10")
OUTPUT_MAP_CMAPS = {"t": "RdYlBu_r", "q": "YlGnBu"}


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
	parser.add_argument("--other_stations_denied", action=argparse.BooleanOptionalAction, default=False,
						help="Remove ALL other stations from the model input (every hour), so the run compares the requested "
							 "station alone present vs absent, instead of vs all other stations present (default off).")
	parser.add_argument("--no_full_output_map_plots", action="store_true",
						help="Skip the full-field maps of the model output (with/without the station), RTMA ges and "
							 "RTMA anl (output_map_*_<var>.png), which are made by default.")
	parser.add_argument("--slice_plots_include_ges", action=argparse.BooleanOptionalAction, default=True,
						help="Add the RTMA background (ges) to the top panel of the lat/lon slice plots "
							 "(--no-slice_plots_include_ges to turn off).")
	parser.add_argument("--slice_plots_include_anl", action=argparse.BooleanOptionalAction, default=True,
						help="Add the RTMA analysis to the top panel of the lat/lon slice plots "
							 "(--no-slice_plots_include_anl to turn off).")
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


def print_neighbourhood(field, lat, lon, y, x, half_width=NEIGHBOURHOOD_HALF_WIDTH):
	"""Print the (2*half_width+1)^2 block of field around grid point (y, x), north at the top and west at the left;
	the station's own value is in brackets."""
	ys = np.arange(max(y - half_width, 0), min(y + half_width + 1, field.shape[0]))
	xs = np.arange(max(x - half_width, 0), min(x + half_width + 1, field.shape[1]))
	if lat[ys[-1], x] < lat[ys[0], x]:
		ys = ys[::-1]
	if lon[y, xs[-1]] < lon[y, xs[0]]:
		xs = xs[::-1]
	print(f"{'':>16}" + "".join(f"{f'x={xx}':>10}" for xx in xs) + "   (west -> east)")
	for yy in ys:
		cells = "".join(f"{f'[{field[yy, xx]:.3f}]' if (yy, xx) == (y, x) else f'{field[yy, xx]:.3f}':>10}" for xx in xs)
		print(f"{f'y={yy}':>16}{cells}")


class _Tee:
	"""File-like object that writes to several streams (here the console and the stats_output.txt log)."""

	def __init__(self, *streams):
		self.streams = streams

	def write(self, text):
		for stream in self.streams:
			stream.write(text)
		return len(text)

	def flush(self):
		for stream in self.streams:
			stream.flush()


def denial_output_dir(args):
	"""Folder for one denial case: <model>/station_denial/station_x_<x>_y_<y>/<YYYY-MM-DD_HH>/other_stations_<denied|included>."""
	y, x = args.station
	others = "denied" if args.other_stations_denied else "included"
	return CASE_DIR / args.model / "station_denial" / f"station_x_{x}_y_{y}" / f"{args.time:%Y-%m-%d_%H}" / f"other_stations_{others}"


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
	only_station = (y, x) if args.other_stations_denied else None
	full = run_model_inference(model, input_file, params, args.stats_path, device, only_station=only_station)
	denied = run_model_inference(model, input_file, params, args.stats_path, device,
								 deny_station=(y, x), deny_station_hours=hours, only_station=only_station)
	n_changed = np.count_nonzero(full["input_array_norm"] != denied["input_array_norm"])
	if n_changed == 0:
		raise RuntimeError("Denial did not change the model input.")

	selected = {"y": y, "x": x, "lat": float(lat[y, x]), "lon": float(lon[y, x])}
	stations = load_stations(path)
	stations["held_out"] = None
	dist = haversine_km(selected["lat"], selected["lon"], lat, lon)
	hour_text = "all hours" if len(hours) == n_window else "hour(s) " + ", ".join(map(str, hours))
	others_text = "all other stations denied" if args.other_stations_denied else "all other stations present"
	print(f"Station ({y}, {x}) at {selected['lat']:.4f}, {selected['lon']:.4f} denied for {hour_text} "
		  f"(reports at hour(s) {reported}; 0 = analysis hour); {others_text}; {n_changed} input values changed.")
	print(f"{'var':>4} {'obs':>10} {'with':>10} {'without':>10} {'diff@stn':>10} {'max|diff|':>10} {'at_km':>8}")

	names = full["output_channel_names"]
	out_dir = denial_output_dir(args)
	print(f"Output folder: {out_dir}")
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
					f"random holdout off  |  {others_text}\n"
					f"At station: obs {obs:.4g}, with {with_station[y, x]:.4g}, without {without_station[y, x]:.4g} "
					f"{units}  |  max |diff| {abs_diff[peak]:.3g} ({dist[peak]:.1f} km from station)")
		out_path = str(out_dir / DENIAL_MAP_NAME.format(var=var))
		plot_denial_map(diff, lat, lon, stations, selected, args.radius_km, out_path, title, subtitle,
						f"With minus without station [{units}]")
		saved.append(out_path)

		# Innovation maps: each model run minus the RTMA analysis (both already in g/kg for q)
		rtma_anl = full["target_analysis_unnorm"][channel]
		in_map = dist <= args.radius_km
		# one colour limit for both innovation maps of this variable so they compare directly
		innov_vmax = denial_map_vmax([with_station - rtma_anl, without_station - rtma_anl], lat, lon, selected,
									 args.radius_km)
		for which, model_field in (("with", with_station), ("without", without_station)):
			innovation = model_field - rtma_anl
			abs_innov = np.where(in_map, np.abs(innovation), -np.inf)
			innov_peak = np.unravel_index(np.argmax(abs_innov), innovation.shape)
			print(f"{'':>4} innovation ({which} station) at station {innovation[y, x]:>9.3g}; "
				  f"max |innovation| within {args.radius_km:g} km {abs_innov[innov_peak]:.3g} ({dist[innov_peak]:.1f} km away); "
				  f"RTMA analysis at station {rtma_anl[y, x]:.4g}")
			print(f"{'':>4} innovation {units}, {NEIGHBOURHOOD_HALF_WIDTH * 2 + 1}x{NEIGHBOURHOOD_HALF_WIDTH * 2 + 1} "
				  f"around the station ({var}, {which} station; north at top, station in brackets):")
			print_neighbourhood(innovation, lat, lon, y, x)
			state = "With station" if which == "with" else f"Station denied ({hour_text})"
			innov_title = (f"Model innovation vs RTMA analysis, {var}: ({y}, {x})  |  "
						   f"lat {selected['lat']:.4f}, lon {selected['lon']:.4f}")
			innov_subtitle = (f"{args.model}  |  {args.time:%Y-%m-%d_%H}  |  {state.lower()} (0 = analysis hour)  |  "
							  f"random holdout off  |  {others_text}\n"
							  f"At station: obs {obs:.4g}, model {model_field[y, x]:.4g}, RTMA analysis {rtma_anl[y, x]:.4g}, "
							  f"innovation {innovation[y, x]:.3g} {units}  |  max |innovation| within {args.radius_km:g} km "
							  f"{abs_innov[innov_peak]:.3g} ({dist[innov_peak]:.1f} km from station)")
			innov_path = str(out_dir / INNOVATION_MAP_NAME.format(which=which, var=var))
			plot_denial_map(innovation, lat, lon, stations, selected, args.radius_km, innov_path, innov_title,
							innov_subtitle, f"Model ({which} station) minus RTMA analysis [{units}]",
							station_label="Station" if which == "with" else "Denied station", title2=state,
							vmax=innov_vmax)
			saved.append(innov_path)

		# RTMA first guess; it comes back unscaled (q in kg/kg) while the analysis arrays are already in g/kg
		ges_field = full["inp_pred_unnorm"][params.inp_pred_vars.index(f"rtma_ges_{var}")] * CASE_SCALE.get(var, 1.0)

		if not args.no_full_output_map_plots:
			# full-field maps of the model run with and without the station, RTMA ges and RTMA anl; the four of a
			# variable share one colour range so they compare directly
			fields = {"model_with_station": with_station, "model_without_station": without_station,
					  "ges": ges_field, "anl": rtma_anl}
			if var in SIGNED_VARS:
				vmax_field = denial_map_vmax(list(fields.values()), lat, lon, selected, args.radius_km)
				vmin_field, cmap = -vmax_field, "RdBu_r"
			else:
				vmin_field, vmax_field = field_map_limits(list(fields.values()), lat, lon, selected, args.radius_km)
				cmap = OUTPUT_MAP_CMAPS[var]
			field_units = CASE_UNITS[var]
			field_names = {"model_with_station": "Model output", "model_without_station": "Model output",
						   "ges": "RTMA first guess (ges)", "anl": "RTMA analysis (anl)"}
			# second title line and station status of each map (the RTMA fields do not depend on the station)
			field_states = {"model_with_station": "With station",
							"model_without_station": f"Station denied ({hour_text})"}
			target_status = {"model_with_station": "included", "model_without_station": "denied"}
			values_text = (f"model with {with_station[y, x]:.4g}, model without {without_station[y, x]:.4g}, "
						   f"RTMA ges {ges_field[y, x]:.4g}, RTMA anl {rtma_anl[y, x]:.4g}")
			for which, field in fields.items():
				low, high = field_map_limits([field], lat, lon, selected, args.radius_km)
				field_title = (f"{field_names[which]}, {var}: ({y}, {x})  |  "
							   f"lat {selected['lat']:.4f}, lon {selected['lon']:.4f}")
				station_text = (f"  |  target station ({y}, {x}) {target_status[which]}  |  {others_text}"
								if which in target_status else "")
				field_subtitle = (f"{args.model}  |  {args.time:%Y-%m-%d_%H}  |  random holdout off"
								  + station_text + "\n"
								  f"At station: obs {obs:.4g}, {values_text} {field_units}  |  "
								  f"in view: min {low:.4g}, max {high:.4g}")
				field_path = str(out_dir / OUTPUT_MAP_NAME.format(which=which, var=var))
				plot_denial_map(field, lat, lon, stations, selected, args.radius_km, field_path, field_title,
								field_subtitle, f"{field_names[which]} [{field_units}]",
								station_label="Denied station" if which == "model_without_station" else "Station",
								title2=field_states.get(which), vmax=vmax_field, vmin=vmin_field, cmap=cmap)
				saved.append(field_path)

		slice_paths = {along: str(out_dir / DENIAL_SLICE_NAME.format(along=along, var=var)) for along in ("lat", "lon")}
		ges = ges_field if args.slice_plots_include_ges else None
		anl = rtma_anl if args.slice_plots_include_anl else None
		saved += plot_denial_slices(with_station, without_station, lat, lon, selected, obs, args.radius_km,
									slice_paths, title, subtitle, units, stations, ges=ges, anl=anl).values()
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
	# keep a copy of the console output (innovations etc.) next to the plots
	out_dir = denial_output_dir(args)
	out_dir.mkdir(parents=True, exist_ok=True)
	log_path = out_dir / STATS_OUTPUT_NAME
	with open(log_path, "w") as log, contextlib.redirect_stdout(_Tee(sys.stdout, log)):
		print("Command: " + " ".join(["python", *sys.argv]))
		station_denial(args, path)
	print(f"Saved {log_path}")


if __name__ == "__main__":
	main()
