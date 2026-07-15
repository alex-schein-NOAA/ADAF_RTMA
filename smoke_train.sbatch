#!/bin/bash
#SBATCH --job-name=adaf_smoke
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err
#
# Smoke test for the obs-dropout retrain: 2 epochs on a 12-file subset, both batch
# assembly paths. What to check in the log:
#   - loss decreases between epoch 1 and 2
#   - "Valid held-out MSE (mean)=... [q=.. t=.. u10=.. v10=..] over N held-out cells"
#     appears EVERY epoch, with N > 0
#   - "Held-out MSE improved from ..." -> best_ckpt is selected on the validation
#     metric, not on training loss
#   - no dataloader worker crash from the new heldout_mask return element
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
S=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/regen_stage/smoke   # symlinks into data_blosc_combined
# CONFIG picks the architecture: params_default (flat SwinIR) or params_multiscale
# (the U-Net pyramid). OUT follows it so the two smokes don't overwrite each other.
CONFIG="${CONFIG:-./config/params_default.yaml}"
OUT="${OUT:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/exp/dropout_r1/smoke}"
CLONE_ENV=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment  # NOT _pip: lacks ruamel, and compile needs cuda.h on CPATH

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate "$CLONE_ENV"
export PATH="${CLONE_ENV}/bin:$PATH"
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p logs "$OUT/training_checkpoints"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29517

# GPU_ASSEMBLE=False exercises the legacy CPU-assembled path (same new return tuple).
GPU_ASSEMBLE="${GPU_ASSEMBLE:-True}"
# BS: set to the real training batch_size (2) to smoke-test the memory ceiling on the
# actual training path -- compile + DDP + dataloader -- not just a synthetic probe.
BS="${BS:-1}"

echo "==== smoke train start $(date) host=$(hostname) config=${CONFIG} bs=${BS} gpu_assemble=${GPU_ASSEMBLE} ===="
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --config_filepath "$CONFIG" \
    --train_data_path "$S/train_data" \
    --valid_data_path "$S/valid_data" \
    --experiment_dir "$OUT" \
    --checkpoint_path "$OUT/training_checkpoints/ckpt.tar" \
    --best_checkpoint_path "$OUT/training_checkpoints/best_ckpt.tar" \
    --gpu_assemble "$GPU_ASSEMBLE" \
    --max_epochs 2 \
    --batch_size "$BS" \
    --num_data_workers 2 \
    --prefetch_factor 2 \
    --valid_frequency 1 \
    --valid_max_files 0 \
    --save_checkpoint True
rc=$?
echo "==== smoke train done rc=${rc} $(date) ===="
exit $rc
