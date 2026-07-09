#!/bin/bash
#SBATCH --job-name=adaf_inference
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference_%j.out
#
# Stage 1 -- THE model run. inference_parallel.py reads raw test_data/*.nc and
# writes full-grid output/*.nc (base-unit output_/rtma_/hrrr_/sta_ + heldout_mask).
# With --hold_out_obs the outputs feed BOTH plot_inference.py (maps) and
# heldout_eval.py (skill), so the model never runs twice. One month (~676 files)
# is ~75 min on one H100. Parameterize per checkpoint / month:
#
#   CKPT=.../16657218_resume_16705162/best_ckpt.tar \
#   OUTDIR=inference_out/jan2023_e243 GLOB='2023-01-*.nc' sbatch inference.sbatch
#
set -euo pipefail
cd /scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate /scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment
mkdir -p logs
export BLOSC_NTHREADS=1 HDF5_USE_FILE_LOCKING=FALSE

PY=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment/bin/python
CONFIG=/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/config/params_default.yaml
STATS=/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation/stats.csv
DATADIR="${DATADIR:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/heldout_jan2023/test_data}"
CKPT="${CKPT:?set CKPT=/path/to/ckpt.tar}"
OUTDIR="${OUTDIR:?set OUTDIR=inference_out/<run>}"
GLOB="${GLOB:-2023-01-*.nc}"
SEED="${SEED:-0}"

$PY inference_parallel.py \
  --config_filepath "$CONFIG" --stats_path "$STATS" \
  --checkpoint_path "$CKPT" \
  --input_dir "$DATADIR" --output_dir "$OUTDIR" \
  --glob_pattern "$GLOB" \
  --batch_size 4 --num_workers 6 --prefetch_factor 2 \
  --hold_out_obs true --hold_out_obs_ratio 0.1 --obs_mask_seed "$SEED" \
  --write_residual_fields --overwrite
