r"""Statistical evaluation of ADAF inference output for one or more models.

Example (run from ADAF_RTMA with ADAF_environment active):

	python stats_eval.py \
		--models lowres_21563899_resume_22722533 lowres_21563899_resume_22722533_hold_out_obs lowres_EXP_22743744_resume_23081675 lowres_EXP_22743744_resume_23081675_hold_out_obs \
		--num_workers 4

Stats are restricted to the times present in every requested model; hours missing inside that
window are listed in stats_results/missing_times_[start]_[end].txt.
Per model, writes stats_results/[model]_[start]_[end].nc containing rmse_<var>(time),
mae_<var>(y, x), bias_<var>(y, x), and the scalar n_samples; also writes one RMSE time-series PNG per variable
and bar_chart_{rmse,bias}_[start]_[end].png (whole-period, whole-domain RMSE and bias vs RTMA analysis per model).
If any model ends in _hold_out_obs, also writes stats_results/heldout_stats_[start]_[end].nc and
heldout_rmse_timeseries_<var>_[start]_[end].png: RMSE of every model at the stations held out in all
_hold_out_obs models, against station obs and against the RTMA analysis. The same file holds RMSE vs
distance from those stations (0-100 km, 2.5 km rings), plotted as heldout_radial_{obs,anl}_<var>_[start]_[end].png.
Existing stats files are reused unless --overwrite is given.
"""

import argparse
import os
from pathlib import Path

from utils.stats_functions import (
	DEFAULT_DATA_ROOT,
	HOLDOUT_SUFFIX,
	VARIABLES,
	get_or_compute_heldout_stats,
	get_or_compute_model_stats,
	parse_hour,
	plot_bar_charts,
	plot_heldout_radial,
	plot_heldout_rmse_timeseries,
	plot_rmse_timeseries,
	restrict_to_common_times,
	write_missing_times_log,
)


def get_args():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--models", nargs="+", required=True, help="Model subfolder names under --data_root.")
	parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
	parser.add_argument("--output_dir", default=str(Path(__file__).parent / "stats_results"))
	parser.add_argument("--variables", nargs="+", default=list(VARIABLES), choices=VARIABLES)
	parser.add_argument("--start_time", type=parse_hour, default=None, help="Inclusive, YYYY-MM-DD_HH.")
	parser.add_argument("--end_time", type=parse_hour, default=None, help="Inclusive, YYYY-MM-DD_HH.")
	parser.add_argument("--num_workers", type=int, default=1)
	parser.add_argument("--overwrite", action="store_true", help="Recompute even if a stats file exists.")
	return parser.parse_args()


def main():
	args = get_args()
	model_dirs = [os.path.join(args.data_root, model) for model in args.models]
	times, model_times = restrict_to_common_times(model_dirs, args.start_time, args.end_time)
	write_missing_times_log(times, model_times, args.output_dir)
	model_stats = {}
	for model_dir in model_dirs:
		name, ds = get_or_compute_model_stats(
			model_dir, times, args.output_dir, overwrite=args.overwrite,
			variables=tuple(args.variables), num_workers=args.num_workers,
		)
		model_stats[name] = ds
	for path in plot_rmse_timeseries(model_stats, args.variables, args.output_dir):
		print(f"Saved {path}")
	for path in plot_bar_charts(model_stats, args.variables, args.output_dir):
		print(f"Saved {path}")

	if not any(os.path.normpath(d).endswith(HOLDOUT_SUFFIX) for d in model_dirs):
		print(f"No {HOLDOUT_SUFFIX} model requested; skipping held-out station stats.")
		return
	heldout = get_or_compute_heldout_stats(
		model_dirs, times, args.output_dir, overwrite=args.overwrite,
		variables=tuple(args.variables), num_workers=args.num_workers,
	)
	for path in plot_heldout_rmse_timeseries(heldout, args.variables, args.output_dir, show_anl_panel=True):
		print(f"Saved {path}")
	for path in plot_heldout_radial(heldout, args.variables, args.output_dir):
		print(f"Saved {path}")


if __name__ == "__main__":
	main()
