#!/bin/bash
#SBATCH --job-name=adaf_lowres_r1
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00
# 12h, not 48: the gpu-wizard account has a MaxGRESRunMinsPerAccount cap. Alongside a
# concurrent 2-GPU job (dropout_r1), 4 GPU x 48h AND 4 GPU x 24h both queued for ~9h
# behind the cap; 4 GPU x 12h = 48 GPU-h fit under it and started IMMEDIATELY. At
# ~850 s/epoch (4 GPU) that is ~50 epochs -- past dropout_r1's e34 reference and the e10
# kill check, though with less margin than 24h. If more epochs are needed, the run
# checkpoints every epoch and can be resumed (mind the resume caveats: best_valid_metric
# resets to 1e6, and lr is force-reset to params.lr -- snapshot best_ckpt first). Raise
# the walltime only when the account is otherwise idle.
#SBATCH --export=ALL
#SBATCH -o logs/lowres_r1_%j.out
#SBATCH -e logs/lowres_r1_%j.err
#
# 1/4-res Swin body, round 1 (HANDOFF_lowres_arch.md). Trains FROM SCRATCH -- different
# parameter set, so it cannot warm-start from dropout_r1, multiscale_r1 or e358.
#
# Override CONFIG/EXP to reuse this for the LR smokes:
#   sbatch --export=ALL,CONFIG=./config/params_lowres_smoke_5e4.yaml,EXP=.../exp/lowres_smoke_5e4 train_lowres.sbatch
#
# The one variable vs dropout_r1 is the ARCHITECTURE: every obs-dropout key,
# loss_channel_weights, obs_mask_seed, valid_max_files and batch_size is identical, so the
# per-epoch HELDOUT_METRICS lines of the two runs are directly comparable. (lr and
# warmup_iters do differ -- 2E-3 diverges on a multi-M-param transformer -- which is an
# unavoidable confound the pyramid run also carried.)
#
# What changed vs the pyramid, which did not learn (35 epochs, held-out temperature pinned
# at HRRR's own value): all attention now runs at 1/4 res, every resampling conv is
# normalized + activated instead of bare and linear, the full-res skip is fused with a 3x3
# instead of a 1x1 (a 1x1 has zero spatial extent and cannot spread an obs at all), and
# conv_after_body is LayerScale-gated so the body cannot simply be abandoned in favour of
# the skip. Measured: 132 cells of reach (flat: 12), 1.98 M params, 28.0 GiB at bs=2 (flat:
# 53.2), 144 ms/sample (flat: 699).
#
# TWO nodes x 2 H100 = 4 ranks, bs=1 -> effective batch 4 (== dropout_r1's 1-node x 2 x
# bs=2), so the A/B stays clean and lr is unchanged; ~1.5x faster wall clock than 1 node.
# Multi-node DDP step time swings ~2x run-to-run on the shared fabric, but this is a
# TRAINING run, not a timing benchmark, so the jitter only makes ms/step noisy -- the
# science is unaffected. To go back to 1 node: set --nodes=1 above and batch_size: 2 in
# the config (effective batch stays 4), and the srun/torchrun block below still works
# (NNODES=1). The all-reduce is tiny (1.98 M params ~= 8 MB), so cross-node comms is cheap.
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
CONFIG="${CONFIG:-./config/params_lowres.yaml}"
EXP="${EXP:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/exp/lowres_r1}"
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
export MASTER_PORT=29520  # NOT 29519: multiscale_r1 uses that, and a concurrent job would collide
NNODES=${SLURM_NNODES:-1}

echo "==== lowres_r1 start $(date) nodes=${NNODES} (${SLURM_JOB_NODELIST}) master=${MASTER_ADDR} job=${SLURM_JOB_ID} config=${CONFIG} exp=${EXP} ===="
# One torchrun per node (srun --ntasks-per-node=1 from the SBATCH header); each spawns 2
# procs. --node_rank comes from SLURM_NODEID so the 4 ranks number 0..3 across both nodes.
srun --kill-on-bad-exit=1 bash -c "
torchrun \
    --nnodes=${NNODES} --nproc_per_node=2 \
    --node_rank=\${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py --config_filepath '${CONFIG}'
"
rc=$?
echo "==== lowres_r1 done rc=${rc} $(date) ===="
exit $rc
