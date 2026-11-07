r"""Statistical evaluation of ADAF inference output for one or more models.

Example (run from ADAF_RTMA with ADAF_environment active):

	python stats_bulk_eval.py \
		--models lowres_21563899_resume_22722533 lowres_21563899_resume_22722533_hold_out_obs lowres_EXP_22743744_resume_23412634 lowres_EXP_22743744_resume_23412634_hold_out_obs lowres_EXP_24482307_resume_25291934 lowres_EXP_24482307_resume_25291934_hold_out_obs \
			--num_workers 16 --start_time "2023-01-01_00" --end_time "2023-12-31_23" --output_dir stats_results/2023-01-01_00_2023-12-31_23

Stats are restricted to the times present in every requested model; hours missing inside that
window are listed in <output_dir>/missing_times_[start]_[end].txt.
Per model, writes <output_dir>/[model]_[start]_[end].nc containing rmse_<var>(time),
mae_<var>(y, x), bias_<var>(y, x), and the scalar n_samples (these stats files and the log stay in <output_dir>).
If any model ends in _hold_out_obs, also writes <output_dir>/heldout_stats_[start]_[end].nc; it holds RMSE of every
model at the stations held out in all _hold_out_obs models, against station obs and against the RTMA analysis, and
RMSE vs distance from those stations (0-100 km, 2.5 km rings).
Existing stats files are reused unless --overwrite is given.

The plots are sorted into subfolders of <output_dir> (created automatically):
	bar_charts/                bar_chart_{rmse,bias}_[start]_[end].png (whole-period, whole-domain RMSE and bias vs
	                           RTMA analysis per model), bar_chart_correlation_{ges,anl}_<var>.png and
	                           bar_chart_correlation_heldout_obs_<var>.png (see below)
	rmse_timeseries_plots/     rmse_timeseries_<var>_[start]_[end].png and heldout_rmse_timeseries_<var>_[start]_[end].png
	heldout_radial_plots/      heldout_radial_{obs,anl}_<var>_[start]_[end].png

Correlation bar charts (skip with --no_correlation_plots): bar_chart_correlation_ges_<var>.png has a box per model for
the Pearson correlation of the model output with the RTMA first guess, plus a last box for the RTMA analysis against the
first guess; bar_chart_correlation_anl_<var>.png is the same against the analysis (last box: first guess). The
correlation is over ALL pixels at ALL hours (one overall mean per field). It needs one extra pass over every hourly
file of every model (this is the slow part; the per-hour sums are cached in <output_dir>/correlation_<model>_[start]_[end].nc
so adding a model only computes the new one). Each box is the block bootstrap distribution of that correlation (box 25-75 %,
whiskers 2.5-97.5 %, diamond = full-sample value; --corr_n_boot resamples of --corr_block_days-day blocks).

Held-out station correlation (needs a _hold_out_obs model; skip with --no_heldout_correlation_plots):
bar_chart_correlation_heldout_obs_<var>.png has the same layout, with a box per model plus RTMA ges and RTMA anl, for the
Pearson correlation with the station observations at the stations held out in all _hold_out_obs models (pooled over all
held-out station-hours; same bootstrap). The per-hour sums are cached in <output_dir>/correlation_heldout_obs_[start]_[end].nc.
"""

import argparse
import os
from pathlib import Path

from utils.stats_functions import (
	DEFAULT_DATA_ROOT,
	HOLDOUT_SUFFIX,
	VARIABLES,
	correlation_bootstrap,
	get_or_compute_correlation_stats,
	get_or_compute_heldout_correlation_stats,
	get_or_compute_heldout_stats,
	get_or_compute_model_stats,
	heldout_correlation_bootstrap,
	parse_hour,
	plot_bar_charts,
	plot_correlation_bar_charts,
	plot_heldout_correlation_bar_charts,
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
	parser.add_argument("--no_correlation_plots", action="store_true",
						help="Skip the all-pixel correlation bar charts (they need one extra pass over every hourly file of every model).")
	parser.add_argument("--no_heldout_correlation_plots", action="store_true",
						help="Skip the correlation-with-held-out-station-obs bar charts (one extra pass over the files; "
							 "only made if a _hold_out_obs model is requested).")
	parser.add_argument("--corr_n_boot", type=int, default=10000, help="Block-bootstrap resamples for the correlation boxes.")
	parser.add_argument("--corr_block_days", type=int, default=1,
						help="Length of the resampled blocks in days (longer blocks allow for day-to-day dependence).")
	parser.add_argument("--corr_seed", type=int, default=0, help="Random seed of the bootstrap.")
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

	if not args.no_correlation_plots:
		corr_stats = {}
		for model_dir in model_dirs:
			name, ds = get_or_compute_correlation_stats(
				model_dir, times, args.output_dir, overwrite=args.overwrite,
				variables=tuple(args.variables), num_workers=args.num_workers,
			)
			corr_stats[name] = ds
		boot = correlation_bootstrap(corr_stats, n_boot=args.corr_n_boot, block_days=args.corr_block_days, seed=args.corr_seed)
		for path in plot_correlation_bar_charts(boot, args.variables, args.output_dir):
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

	if not args.no_heldout_correlation_plots:
		heldout_corr = get_or_compute_heldout_correlation_stats(
			model_dirs, times, args.output_dir, overwrite=args.overwrite,
			variables=tuple(args.variables), num_workers=args.num_workers,
		)
		boot = heldout_correlation_bootstrap(heldout_corr, n_boot=args.corr_n_boot, block_days=args.corr_block_days, seed=args.corr_seed)
		for path in plot_heldout_correlation_bar_charts(boot, args.variables, args.output_dir):
			print(f"Saved {path}")


if __name__ == "__main__":
	main()
