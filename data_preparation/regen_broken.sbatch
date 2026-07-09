#!/bin/bash
#SBATCH --job-name=adaf_regen
#SBATCH --account=wrfruc
#SBATCH --qos=batch
#SBATCH --partition=u1-compute
#SBATCH --array=1-40
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/regen_%A_%a.out
#
# Regenerate the cycles broken by the ceil('h') obs-binning bug (4th obs_time_window
# bin one hour past analysis time). Writes Blosc-ZSTD-L3 straight into
# data_blosc_combined/, matching convert_blosc.py. heldout_jan2023/test_data is
# symlinks into that dir, so it is fixed for free.
#
# ~2323 cycles, ~55 s each => ~35 CPU-hours; 40 shards ~= 55 min/shard.
#
#   sbatch data_preparation/regen_broken.sbatch
#
# Dry-run into a staging dir first (nothing is overwritten):
#   REGEN_DST=/scratch3/.../regen_stage/staging sbatch data_preparation/regen_broken.sbatch
#
# The originals are NOT backed up because they are reproducible: aschein's untouched
# zlib copies live at /scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/.
set -uo pipefail

CLONE_ENV="/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment"
PREP="/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA/data_preparation"
LIST="${REGEN_LIST:-${PREP}/broken_cycles.txt}"
DST="${REGEN_DST:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/data_blosc_combined}"
WORK="/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/regen_stage/work/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate "$CLONE_ENV"
export PATH="${CLONE_ENV}/bin:$PATH"
export HDF5_USE_FILE_LOCKING=FALSE
export BLOSC_NTHREADS=1
export OMP_NUM_THREADS=1

mkdir -p "$WORK"
echo "==== regen shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} host=$(hostname) start=$(date) ===="
echo "list=${LIST}  dst=${DST}"

python "${PREP}/regen_broken_cycles.py" \
    --list "$LIST" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --workdir "$WORK" \
    --dst-root "$DST"
rc=$?

rmdir "$WORK" 2>/dev/null
echo "==== regen shard ${SLURM_ARRAY_TASK_ID} done rc=${rc} $(date) ===="
exit $rc
