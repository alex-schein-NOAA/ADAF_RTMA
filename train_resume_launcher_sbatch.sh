#!/bin/bash
#SBATCH --account gpu-ghpcs
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH -J ADAF_resume
#SBATCH -o TEST_JOB_LOGS/ADAF_resume_%J.out
#SBATCH -e TEST_JOB_LOGS/ADAF_resume_%J.err

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # BACK TO: one launcher task per node
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2                 # 2 GPUs per node
#SBATCH --mem=0
# NO --gpus-per-task - let all GPUs be visible to the launcher task

#SBATCH -t 01:30:00
#SBATCH --export=ALL

echo "Starting job"

# --- Threading: 2 ranks × 2 threads = 4 CPUs/node ---
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# --- NCCL / rendezvous ---
#export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Network (tweak iface if needed, e.g., ib0, enp175s0f0np0)
# export NCCL_SOCKET_IFNAME=^lo,docker0

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
conda activate ADAF_environment_pip

echo "After Python load: CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

###############
# MUST fill this out with the job number of the run you want to resume from
MODEL_NUMBER_TO_LOAD=0000
PREVIOUS_CHECKPOINT_DIR="/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/${MODEL_NUMBER_TO_LOAD}" #Modify this if resuming from a resumed job

CHECKPOINT_DIR="/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/${MODEL_NUMBER_TO_LOAD}_resume_${SLURM_JOB_ID}"
mkdir -p "${CHECKPOINT_DIR}"

# --- Quick sanity check on *every* node about GPU visibility/binding ---
#NOT adding the arguments from Raj's code, at least not yet
srun --ntasks-per-node=2 --mpi=none \
     --gres=gpu:2 \
     bash -lc 'echo "Host: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true'

# --- Launch: one torchrun per node; each spawns 2 ranks (1 per GPU) ---

srun --ntasks-per-node=1 --mpi=none \
     --gres=gpu:2 \
    python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node=2 \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend="${RDZV_BACKEND}" \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --rdzv_id="${RDZV_ID}" \
     /scratch3/BMC/wrfruc/aschein/ADAF_new/train.py \
     --config_filepath "./config/params_default.yaml" \
     --resuming True \
     --max_epochs 20 \
     --resume_checkpoint_path "${PREVIOUS_CHECKPOINT_DIR}/best_ckpt.tar" \
     --checkpoint_path "${CHECKPOINT_DIR}/ckpt.tar" \
     --best_checkpoint_path "${CHECKPOINT_DIR}/best_ckpt.tar"

##### Notes:
# - max_epochs here must be greater than the number of epochs when the previous best checkpoint was saved, as the training will resume from the epoch the model was saved at and continue to max_epochs
# - resume_checkpoint_path can also load ckpt.tar if desired
# - If resuming from a job that was itself a resumed job, just modify PREVIOUS_CHECKPOINT_DIR to point to the correct path; MODEL_NUMBER_TO_LOAD will remain the same as this is the base job that all resumed jobs spring from

stopTime=$(date +%s)
echo "runTime=$((stopTime-startTime))"