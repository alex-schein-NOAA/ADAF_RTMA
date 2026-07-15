#!/bin/bash
#SBATCH --job-name=adaf_heldout_eval
#SBATCH --account=wrfruc
#SBATCH --qos=batch
#SBATCH --partition=u1-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/heldout_eval_%j.out
#
# Stage 2 -- NO GPU, NO torch. heldout_eval.py reads the Stage-1 output/*.nc,
# scores the held-out obs (Model / HRRR / RTMA), caches per-point records, and
# writes the 7 skill PNGs. I/O-bound: a full month (741 files) takes well over
# 10 min, so submit it rather than running on the login node. Parameterize:
#
#   INDIR=inference_out/jan2023_e243 OUTDIR=Plots/jan2023_e243 \
#   GLOB='2023-01-*.nc' sbatch heldout_eval.sbatch
#
# Re-style from the cache with no job at all:
#   python heldout_eval.py --from-cache Plots/jan2023_e243/records.npz \
#       --outdir Plots/jan2023_e243
#
set -euo pipefail
cd /scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate /scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment
mkdir -p logs
export HDF5_USE_FILE_LOCKING=FALSE

PY=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment/bin/python
INDIR="${INDIR:?set INDIR=inference_out/<run>}"
OUTDIR="${OUTDIR:?set OUTDIR=Plots/<run>}"
GLOB="${GLOB:-*.nc}"

$PY heldout_eval.py \
  --indir "$INDIR" --glob "$GLOB" \
  --outdir "$OUTDIR" --cache "$OUTDIR/records.npz"
