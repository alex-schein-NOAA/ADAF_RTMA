#!/bin/bash
#SBATCH --account gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH -J ADAF_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH --export=ALL
#SBATCH -o logs/ADAF_train_%j.out
#SBATCH -e logs/ADAF_train_%j.err
#
# Production training: train.py under DDP (2 nodes x 2 H100 = 4 ranks) with the
# speedups from SPEEDUP_CHANGELOG.md (torch.compile + bf16 + channels_last + tf32 +
# ddp_broadcast_buffers=False). All of those live in config/params_default.yaml.
#
# REQUIRES the CLONED env (its activate.d hook puts cuda.h on CPATH so torch.compile
# works); the env is activated INSIDE srun and the interpreter is called by ABSOLUTE
# PATH (the multi-node PATH race otherwise falls back to a torch-less module python).
#
# Submit:  sbatch train_production_sbatch.sh
# Resume:  CONFIG=... pass --resuming True via params, or use train_resume_launcher_sbatch.sh
set -euo pipefail

REPO="${ADAF_REPO:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA}"
CLONE_ENV="/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment"
CONFIG="${ADAF_CONFIG:-${REPO}/config/params_default.yaml}"

mkdir -p "${REPO}/logs"

# Threading: 2 ranks x 2 threads per node.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# NCCL / torch.distributed rendezvous (shared across nodes).
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT=${MASTER_ADDR}:${MASTER_PORT}
export RDZV_ID=$SLURM_JOB_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== ADAF production train | config=${CONFIG} ===="
echo "MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES start=$(date)"
cd "${REPO}"

srun --ntasks-per-node=1 --mpi=none --gres=gpu:2 \
    bash -c '
        module load python
        source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
        conda activate "'"${CLONE_ENV}"'"
        # Call the clone interpreter by ABSOLUTE PATH: under multi-node srun the module
        # python can win the PATH race even after conda activate (which still runs the
        # activate.d hook -> cuda.h on CPATH for torch.compile).
        CLONE_PY="'"${CLONE_ENV}"'/bin/python"
        export PATH="'"${CLONE_ENV}"'/bin:$PATH"
        # Avoid "NetCDF: HDF error" from concurrent HDF5 reads on Lustre across workers.
        export HDF5_USE_FILE_LOCKING=FALSE
        echo "node $SLURM_NODEID python: $("$CLONE_PY" -c "import sys;print(sys.executable)") | torch: $("$CLONE_PY" -c "import torch;print(torch.__version__)")"
        exec "$CLONE_PY" -m torch.distributed.run \
            --nnodes="'"${NNODES}"'" --nproc_per_node=2 \
            --node_rank="$SLURM_NODEID" \
            --rdzv_backend="'"${RDZV_BACKEND}"'" \
            --rdzv_endpoint="'"${RDZV_ENDPOINT}"'" \
            --rdzv_id="'"${RDZV_ID}"'" \
            "'"${REPO}"'/train.py" --config_filepath "'"${CONFIG}"'"
    '
echo "==== ADAF production train done $(date) ===="
