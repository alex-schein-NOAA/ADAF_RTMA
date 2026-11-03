r"""Launch stats_case_eval.py (single-station denial case studies) for many models / stations / times in one go.

Runs, one subprocess per case, every combination of MODELS x STATIONS x TIMES x (other stations included, other
stations denied), with --radius_km 120. A case whose output folder already exists under stats_results/case_studies/
(see denial_output_dir in stats_case_eval.py) is skipped, so the wrapper can be restarted after an interruption; the
folder of a case that fails or is interrupted is removed so that the next launch retries it. Cases whose model
inference file for the hour is missing are skipped with a message. The failures are listed at the end (exit code 1).

Before anything runs, the cases that would run are checked for stations that reported no observation at the analysis
hour (the station is absent that hour, so stats_case_eval.py would stop with "no observations at the denied hour(s)").
If there are any, they are listed and you are asked to type y to continue; the continued run skips those cases. Use
--yes to continue without the prompt (e.g. in a batch job; the warning is still printed, and without --yes a batch
job with no input stops there).

Edit MODELS, STATIONS and TIMES below. Stations are grid indices (y, x), the same order as stats_case_eval.py --station Y X.

Needs a GPU, like stats_case_eval.py. From a GPU node, from ADAF_RTMA with ADAF_environment active:

	python stats_case_eval_multicase_wrapper.py
	python stats_case_eval_multicase_wrapper.py --dry_run          # list what would run and what is skipped
	python stats_case_eval_multicase_wrapper.py --yes              # no prompt when stations are absent
	python stats_case_eval_multicase_wrapper.py --models lowres_21563899_resume_22722533   # other models than MODELS

Other arguments (e.g. --device cuda, --input_dir, --config_filepath, --no_full_output_map_plots) are passed on to
stats_case_eval.py unchanged; --radius_km 120 is always given first, so repeating it here would override it.
"""

import argparse
import itertools
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401  registers the Blosc2 filter used by the inference output
import numpy as np

from stats_case_eval import denial_output_dir
from utils.stats_functions import DEFAULT_DATA_ROOT, VARIABLES, model_file_path, parse_hour

PROJECT_DIR = Path(__file__).resolve().parent
CASE_SCRIPT = PROJECT_DIR / "stats_case_eval.py"

MODELS = ["lowres_21563899_resume_22722533", "lowres_EXP_22743744_resume_23412634"]
STATIONS = [(783, 933), (584, 1164), (949, 2098), (952, 516)]  # (y, x)
TIMES = ["2023-03-15_06", "2023-03-15_18", "2023-06-15_06", "2023-06-15_18",
		 "2023-09-15_06", "2023-09-15_18", "2023-12-15_07", "2023-12-15_19"]
RADIUS_KM = 120


def get_args():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--models", nargs="+", default=MODELS, help=f"Model subfolders (default: {MODELS}).")
	parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
	parser.add_argument("--dry_run", action="store_true", help="Only list the cases and whether they would run or be skipped.")
	parser.add_argument("--yes", action="store_true",
						help="Continue without asking when some stations are absent at some hours (the warning is still printed).")
	return parser.parse_known_args()


def raise_interrupt(signum, frame):
	raise KeyboardInterrupt


def find_absent_stations(pending, data_root):
	"""Set of (model, time_text, (y, x)) among the pending cases whose station has no observation at the analysis hour.

	Reads obs_<var>[-1] (the analysis hour; NaN where there is no station) of each model's inference file once and
	looks up every station requested for it. A station counts as absent if any variable is not finite there (data
	prep keeps stations only when all variables are present, so this is all-or-nothing in practice).
	"""
	wanted = {}
	for model, (y, x), time_text, _ in pending:
		wanted.setdefault((model, time_text), set()).add((y, x))
	absent = set()
	for (model, time_text), stations in wanted.items():
		ys, xs = (np.array(index) for index in zip(*sorted(stations)))
		path = model_file_path(os.path.join(data_root, model), parse_hour(time_text))
		with h5py.File(path, "r") as h5:
			present = np.ones(ys.size, dtype=bool)
			for var in VARIABLES:
				present &= np.isfinite(h5[f"obs_{var}"][-1][ys, xs])
		absent |= {(model, time_text, (int(y), int(x))) for y, x, ok in zip(ys, xs, present) if not ok}
	return absent


def main():
	args, extra = get_args()
	for time_text in TIMES:
		parse_hour(time_text)  # fail early on a malformed time
	# a SLURM cancel (SIGTERM) is handled like Ctrl-C so the folder of the running case is cleaned up
	signal.signal(signal.SIGTERM, raise_interrupt)

	cases = list(itertools.product(args.models, STATIONS, TIMES, (False, True)))
	done, existing, missing, absent_cases, failed = [], [], [], [], []

	def describe(model, y, x, time_text, denied):
		return f"{model} station (y={y}, x={x}) {time_text} other_stations_{'denied' if denied else 'included'}"

	# pass 1: which cases would run at all
	pending = []
	for number, case in enumerate(cases, 1):
		model, (y, x), time_text, denied = case
		prefix = f"[{number}/{len(cases)}] {describe(model, y, x, time_text, denied)}"
		hour = parse_hour(time_text)
		out_dir = denial_output_dir(SimpleNamespace(model=model, station=(y, x), time=hour, other_stations_denied=denied))
		if out_dir.exists():
			print(f"{prefix}: SKIP, folder exists ({out_dir})", flush=True)
			existing.append(describe(model, y, x, time_text, denied))
		elif not os.path.exists(model_file_path(os.path.join(args.data_root, model), hour)):
			print(f"{prefix}: SKIP, no inference file for {time_text} in {os.path.join(args.data_root, model)}", flush=True)
			missing.append(describe(model, y, x, time_text, denied))
		else:
			pending.append((number, case, out_dir))

	# pass 2: stations without an observation at the analysis hour would make stats_case_eval.py fail
	absent = find_absent_stations([case for _, case, _ in pending], args.data_root)
	if absent:
		affected = [item for item in pending if (item[1][0], item[1][2], item[1][1]) in absent]
		print(f"\n{'!' * 100}\nWARNING: {len(absent)} station/hour combination(s) have no observation (the station is absent "
			  f"at that analysis hour); {len(affected)} case(s) would fail on them:")
		by_station = {}
		for model, time_text, station in sorted(absent):
			by_station.setdefault((model, station), []).append(time_text)
		for (model, (y, x)), time_texts in by_station.items():
			print(f"  {model}: station (y={y}, x={x}) absent at {', '.join(time_texts)}")
		print(f"Continuing skips these {len(affected)} case(s) and runs the other {len(pending) - len(affected)}.\n{'!' * 100}",
			  flush=True)
		if not args.dry_run and not args.yes:
			try:
				reply = input("Continue? Type y to continue, anything else aborts: ").strip().lower()
			except EOFError:
				reply = ""
				print("\nNo input available (batch job?); use --yes to continue without the prompt.")
			if reply not in ("y", "yes"):
				print("Aborted; nothing was run.")
				sys.exit(1)

	# pass 3: run
	for number, case, out_dir in pending:
		model, (y, x), time_text, denied = case
		label = describe(model, y, x, time_text, denied)
		prefix = f"[{number}/{len(cases)}] {label}"
		if (model, time_text, (y, x)) in absent:
			print(f"{prefix}: SKIP, station absent at the analysis hour", flush=True)
			absent_cases.append(label)
			continue
		command = [sys.executable, str(CASE_SCRIPT), "--model", model, "--data_root", args.data_root,
				   "--time", time_text, "--station", str(y), str(x), "--radius_km", str(RADIUS_KM)]
		if denied:
			command.append("--other_stations_denied")
		command += extra
		if args.dry_run:
			print(f"{prefix}: WOULD RUN {' '.join(command)}", flush=True)
			continue
		print(f"\n{'=' * 100}\n{prefix}: RUN\n{' '.join(command)}\n{'=' * 100}", flush=True)
		try:
			returncode = subprocess.run(command, cwd=PROJECT_DIR).returncode
		except KeyboardInterrupt:
			# subprocess.run has already stopped the child; drop its partial output so a relaunch retries the case
			shutil.rmtree(out_dir, ignore_errors=True)
			print(f"\nInterrupted during {label}; removed {out_dir}. Finished before that: {len(done)}, skipped as "
				  f"existing: {len(existing)}.", flush=True)
			sys.exit(130)
		if returncode == 0:
			done.append(label)
		else:
			shutil.rmtree(out_dir, ignore_errors=True)
			failed.append(f"{label} (exit code {returncode})")
			print(f"{prefix}: FAILED with exit code {returncode}; removed {out_dir} so a relaunch retries it.", flush=True)

	print(f"\n{'=' * 100}\nSummary of {len(cases)} cases"
		  f"{' (dry run)' if args.dry_run else ''}: {len(done)} run, {len(existing)} skipped (folder exists), "
		  f"{len(missing)} skipped (no inference file), {len(absent_cases)} skipped (station absent), {len(failed)} failed")
	for label in failed:
		print(f"  FAILED: {label}")
	for label in missing:
		print(f"  NO FILE: {label}")
	for label in absent_cases:
		print(f"  STATION ABSENT: {label}")
	sys.exit(1 if failed else 0)


if __name__ == "__main__":
	main()
