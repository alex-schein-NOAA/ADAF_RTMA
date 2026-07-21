#!/bin/bash
#SBATCH --job-name=adaf_lowres_r4
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH -o logs/lowres_r4_%j.out
#SBATCH -e logs/lowres_r4_%j.err
#
# lowres ROUND 4: held-out obs IN THE GRADIENT (heldout_loss_weight: 0.1). Single-change
# A/B vs r3 -- see the config header for the diagnosis (memory
# heldout-plateau-is-denial-fixed-point) and the verdict criteria. Trains FROM SCRATCH.
#
# Same 4 nodes x 2 H100 = 8 ranks / bs=1 / lr 7E-4 recipe as r2/r3. Keep the 12h wall:
# the r3 early-stop finding (best by ~e30-40) is a property of the OLD pinned metric --
# with the term in the gradient the metric may keep improving longer.
#
# METAR twin (mirrors the r3/r3_metar pairing):
#   sbatch --export=ALL,CONFIG=./config/params_lowres_r4_metar.yaml,EXP=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/exp/lowres_r4_metar train_lowres_r4.sbatch
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
CONFIG="${CONFIG:-./config/params_lowres_r4.yaml}"
EXP="${EXP:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/exp/lowres_r4}"
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

# Snapshot exactly what produced this run, next to its checkpoints. A checkpoint loaded
# against the wrong config produces garbage silently.
cp "$CONFIG" "$EXP/params_used_${SLURM_JOB_ID}.yaml"
git -C "$REPO" rev-parse HEAD > "$EXP/git_sha_${SLURM_JOB_ID}.txt" 2>/dev/null
git -C "$REPO" diff > "$EXP/git_diff_${SLURM_JOB_ID}.patch" 2>/dev/null

# Rendezvous host = the first node in the allocation; every node's torchrun dials it.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29523  # NOT 29519-29522 (multiscale_r1 / lowres_r1 / r2 / r3): avoid a collision with any concurrent job
NNODES=${SLURM_NNODES:-1}

echo "==== lowres_r4 start $(date) nodes=${NNODES} (${SLURM_JOB_NODELIST}) master=${MASTER_ADDR} job=${SLURM_JOB_ID} config=${CONFIG} exp=${EXP} ===="
# One torchrun per node (srun --ntasks-per-node=1 from the SBATCH header); each spawns 2
# procs; --node_rank from SLURM_NODEID numbers the 8 ranks 0..7 across the 4 nodes.
srun --kill-on-bad-exit=1 bash -c "
torchrun \
    --nnodes=${NNODES} --nproc_per_node=2 \
    --node_rank=\${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py --config_filepath '${CONFIG}' --resuming True --lr 3.5E-4
"
rc=$?
echo "==== lowres_r4 done rc=${rc} $(date) ===="
exit $rc
