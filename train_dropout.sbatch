#!/bin/bash
#SBATCH --job-name=adaf_dropout_r1
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o logs/dropout_r1_%j.out
#SBATCH -e logs/dropout_r1_%j.err
#
# Round 1 of the obs-dropout retrain (HANDOFF_retrain_obs_dropout.md).
# Trains FROM SCRATCH on data_blosc_combined with:
#   - obs dropout ON: 5-30% random station hold-out + 30% chance of a block blackout;
#     held-out stations are hidden from the INPUT but kept in the TARGET, so the model
#     can no longer minimize loss by copying obs straight through from its input.
#   - checkpoint/LR selection on the VALIDATION held-out-station MSE (the deployed skill),
#     not on training loss.
#   - loss_channel_weights [1,1,1,1] -- i.e. today's effective per-variable balance
#     (t 1.0 / q 2.64 / u 0.31 / v 0.42). Deliberately UNCHANGED so this run isolates dropout.
#
# ONE node x 2 H100: multi-node DDP step time swings ~2x run-to-run on the shared
# network, which would make the epoch timings unreadable. ~15 min/epoch -> 100 epochs
# is ~25 h, inside the 48 h limit, so no resume chaining is needed.
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
EXP=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/exp/dropout_r1
CLONE_ENV=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment  # NOT _pip (no ruamel); its activate hook puts cuda.h on CPATH, which torch.compile needs

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate "$CLONE_ENV"
export PATH="${CLONE_ENV}/bin:$PATH"
# Concurrent HDF5 reads on Lustre across workers/ranks otherwise raise "NetCDF: HDF error"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p logs "$EXP/training_checkpoints"

# Snapshot exactly what produced this run, next to its checkpoints.
cp config/params_default.yaml "$EXP/params_used_${SLURM_JOB_ID}.yaml"
git -C "$REPO" rev-parse HEAD > "$EXP/git_sha_${SLURM_JOB_ID}.txt" 2>/dev/null
git -C "$REPO" diff > "$EXP/git_diff_${SLURM_JOB_ID}.patch" 2>/dev/null

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29518

echo "==== dropout_r1 start $(date) host=$(hostname) job=${SLURM_JOB_ID} ===="
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
    --config_filepath ./config/params_default.yaml
rc=$?
echo "==== dropout_r1 done rc=${rc} $(date) ===="
exit $rc
