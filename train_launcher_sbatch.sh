#!/bin/bash
#SBATCH --account gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH -J ADAF_RTMA_train
#SBATCH -o training_runs/lowres_%j/log_%j.out
#SBATCH -e training_runs/lowres_%j/log_%j.err

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # BACK TO: one launcher task per node
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2                 # 2 GPUs per node
#SBATCH --mem=0
# NO --gpus-per-task - let all GPUs be visible to the launcher task

#SBATCH -t 12:00:00 #01:30:00
#SBATCH --export=ALL

echo "Starting job"

# --- Threading: 2 ranks × 2 threads = 4 CPUs/node ---
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Inductor defaults to one compile worker per CPU (192 here), per rank -> host OOM.
export TORCHINDUCTOR_COMPILE_THREADS=8

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

CHECKPOINT_DIR="/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/lowres_${SLURM_JOB_ID}"
mkdir -p "${CHECKPOINT_DIR}"

# --- Stage ptxas + Triton/Inductor caches on node-local disk ---
# Exec'ing ptxas off Lustre from 8 concurrent compile workers gives ETXTBSY.
ENV_BIN=/scratch3/BMC/wrfruc/aschein/miniconda/envs/ADAF_environment/bin
export TRITON_PTXAS_PATH=/tmp/ptxas_${SLURM_JOB_ID}
export TRITON_PTXAS_BLACKWELL_PATH=$TRITON_PTXAS_PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_${SLURM_JOB_ID}

srun --ntasks-per-node=1 --mpi=none bash -lc '
  cp -f '"$ENV_BIN"'/ptxas "$TRITON_PTXAS_PATH".tmp.$$ &&
  mv -f "$TRITON_PTXAS_PATH".tmp.$$ "$TRITON_PTXAS_PATH" &&
  chmod 755 "$TRITON_PTXAS_PATH"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  echo "$(hostname): staged ptxas -> $TRITON_PTXAS_PATH"'

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
     --config_filepath "./config/params_lowres.yaml" \
     --max_epochs 200 \
     --valid_frequency 10 \
     --localsgd_h 50 \
     --target "analysis_obs" \
     --train_sample_fraction 0.5 \
     --checkpoint_path "${CHECKPOINT_DIR}/ckpt.tar" \
     --best_checkpoint_path "${CHECKPOINT_DIR}/best_ckpt.tar"

srun --ntasks-per-node=1 --mpi=none bash -lc '
  rm -rf "$TRITON_PTXAS_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"'

stopTime=$(date +%s)
echo "runTime=$((stopTime-startTime))"
