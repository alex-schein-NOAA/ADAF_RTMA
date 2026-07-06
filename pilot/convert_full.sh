#!/bin/bash
#SBATCH --account gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH -J blosc_convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH -t 6:00:00
#SBATCH --export=ALL
#SBATCH -o /scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/pilot/convert_%x_%j.out
#SBATCH -e /scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/pilot/convert_%x_%j.err
#
# Convert one split (train|valid|test) from zlib /scratch5 -> Blosc-ZSTD-L3 + obs-float32
# on /scratch3. Resumable (skips done files) and verify-after-write (asserts every var
# round-trips exactly). Submit once per split so they run on separate nodes concurrently:
#   sbatch --job-name=cv_train convert_full.sh train
#
# SRC is Alex's LIVE combined mesonet+METAR set (…/aschein/ADAF_new/data, gzip-L1, has
# obs_source label 0/1/2). Splits are by year: train=2021, valid=2022, test=2023. This
# supersedes the old mesonet-only source (…/data_RTMA_grid_Mesonet_stations_only). DST is
# a fresh data_blosc_combined dir so the stale mesonet-only data_blosc stays intact until
# the new one is verified + config repointed.
set -uo pipefail

SPLIT="${1:?usage: convert_full.sh <train|valid|test>}"
NPROCS="${2:-20}"
CLONE_ENV="/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment"
PILOT="/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/pilot"
SRC="/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/${SPLIT}_data"
DST="/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/data_blosc_combined/${SPLIT}_data"

module load python >/dev/null 2>&1
source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate "$CLONE_ENV"
export PATH="${CLONE_ENV}/bin:$PATH"
export HDF5_USE_FILE_LOCKING=FALSE
export BLOSC_NTHREADS=1
export OMP_NUM_THREADS=1

echo "==== convert ${SPLIT} job=${SLURM_JOB_ID} host=$(hostname) start=$(date) ===="
echo "src=${SRC}  ($(ls "$SRC"/*.nc 2>/dev/null | wc -l) files)"
echo "dst=${DST}  nprocs=${NPROCS}"
srun --ntasks=1 python "${PILOT}/convert_blosc.py" "$SRC" "$DST" '*.nc' 0 "$NPROCS" 3
rc=$?
echo "==== convert ${SPLIT} done rc=${rc} $(date) | out=$(ls "$DST"/*.nc 2>/dev/null | wc -l) files, $(du -sh "$DST" 2>/dev/null | cut -f1) ===="
