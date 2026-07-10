#!/bin/bash
#SBATCH --account gpu-esrl-ai
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH -J ADAF_RTMA_train
#SBATCH -o training_runs/%j/log_%j.out
#SBATCH -e training_runs/%j/log_%j.err

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # BACK TO: one launcher task per node
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2                 # 2 GPUs per node
#SBATCH --mem=0
# NO --gpus-per-task - let all GPUs be visible to the launcher task

#SBATCH -t 18:00:00 #01:30:00
#SBATCH --export=ALL

echo "Starting job"

# --- Threading: 2 ranks × 2 threads = 4 CPUs/node ---
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# --- NCCL / rendezvous ---
#export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Rendezvous (shared by all nodes)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT=${MASTER_ADDR}:${MASTER_PORT}
export RDZV_ID=$SLURM_JOB_ID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NODEID=$SLURM_NODEID / SLURM_NNODES=$SLURM_NNODES"

echo "starting at $(date)"
startTime=$(date +%s)
## Added from Raj's code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


###############

echo $PWD

module load python
echo 'Modules loaded'

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh

###############

CHECKPOINT_DIR="/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/${SLURM_JOB_ID}"
mkdir -p "${CHECKPOINT_DIR}"

# --- Quick sanity check on *every* node about GPU visibility/binding ---
srun --ntasks-per-node=2 --mpi=none \
     --gres=gpu:2 \
     bash -lc 'echo "Host: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true'

# --- Launch: one torchrun per node; each spawns 2 ranks (1 per GPU) ---
srun --ntasks-per-node=1 --mpi=none \
     --gres=gpu:2 \
    /scratch3/BMC/wrfruc/aschein/miniconda/envs/ADAF_environment/bin/python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node=2 \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend="${RDZV_BACKEND}" \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --rdzv_id="${RDZV_ID}" \
     /scratch3/BMC/wrfruc/aschein/ADAF_RTMA/train.py \
     --config_filepath "./config/params_default.yaml" \
     --max_epochs 1000 \
     --valid_frequency 10 \
     --localsgd_h 50 \
     --target "analysis" \
     --train_sample_fraction 0.33 \
     --checkpoint_path "${CHECKPOINT_DIR}/ckpt.tar" \
     --best_checkpoint_path "${CHECKPOINT_DIR}/best_ckpt.tar"

stopTime=$(date +%s)
echo "runTime=$((stopTime-startTime))"
