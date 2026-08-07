#!/bin/bash
#SBATCH --account gpu-esrl-ai
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH -J inference_parallel_lowres
#SBATCH -o inference_runs/%j/log_%j.out
#SBATCH -e inference_runs/%j/log_%j.err

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # BACK TO: one launcher task per node
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1                 # 1 GPU per node
#SBATCH --mem=64G

#SBATCH -t 00:15:00 #01:30:00
#SBATCH --export=ALL

echo "starting at $(date)"
startTime=$(date +%s)

if [ "$#" -ne 1 ]; then
    echo "Error: Missing arguments."
    echo "Usage:   $0 ''<year>-<month>-<day>_<hour>.nc'' "
    echo "Example: $0 ''2023-01-01_00.nc'' "
    echo "Can use regex patterns, e.g. ''2023-01-0[1-4]_*.nc''"
    exit 1
fi

#INPUT DATES = USER ARGUMENT
GLOB=$1 #input as needed - should be something like "2023-01-0[1-5]_*.nc"

###############

echo $PWD

module load python
echo 'Modules loaded'

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
unset PYTHONPATH

###############

export BLOSC_NTHREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY=/scratch3/BMC/wrfruc/aschein/miniconda/envs/ADAF_environment/bin/python

CONFIG="${CONFIG:-/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/config/params_lowres.yaml}"
STATS="${STATS:-/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation/stats.csv}"
DATADIR="${DATADIR:-/scratch5/BMC/ai_datadepot/projects/aschein/ADAF_new/data_blosc_combined/test_data}"
CKPT="${CKPT:-/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/lowres_18299112/best_ckpt.tar}" #Change as needed
OUTDIR="${OUTDIR:-/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/inference_outputs/lowres_18299112}" #Change as needed

SEED="${SEED:-1234}"   # hold-out draw seed; -1 = unseeded (irreproducible). Record it with the results.

$PY -u inference_parallel_lowres.py \
  --config_filepath "$CONFIG" --stats_path "$STATS" \
  --checkpoint_path "$CKPT" \
  --input_dir "$DATADIR" --output_dir "$OUTDIR" \
  --glob_pattern "$GLOB" \
  --batch_size 6 --num_workers 6 --prefetch_factor 2 \
  --obs_mask_seed "$SEED" \
  --write_residual_fields --overwrite

stopTime=$(date +%s)
echo "runTime=$((stopTime-startTime))"
