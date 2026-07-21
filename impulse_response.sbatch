#!/bin/bash
#SBATCH --job-name=adaf_impulse
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/impulse_%j.out
#
# 3a -- single-ob point-spread function. ~7 full-grid forward passes on one H100
# (baseline + 2 sites x 3 dT). A few minutes. Writes
# Plots/jan2023_r4_l0p3_e155/architecture/3a_impulse_response.png
set -euo pipefail
cd /scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate /scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment
mkdir -p logs
export BLOSC_NTHREADS=1 HDF5_USE_FILE_LOCKING=FALSE

PY=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment/bin/python
$PY plot_impulse_response.py
