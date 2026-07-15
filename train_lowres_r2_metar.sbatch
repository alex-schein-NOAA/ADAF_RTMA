#!/bin/bash
#SBATCH --job-name=adaf_lowres_r2_metar
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00
# 4 nodes x 2 H100 = 8 GPU (r1 used 2 nodes x 2 = 4). Batch size doesn't change per-sample
# throughput (preflight: 266 ms/sample at bs 1 and 2 alike -- memory-bandwidth-bound), so
# wall-clock speedup comes from RANKS: 8 ranks ~halve the epoch vs 4 (~6 min/epoch est).
# eff batch doubles 4->8 -> lr retuned 5E-4->7E-4 in the config. CAP/QUEUE: gpu-wizard has a
# MaxGRESRunMins-style account cap; 8 GPU x 12h = 96 GPU-h and the account is currently busy
# (another user running 16 GPU), so this will likely QUEUE rather than start immediately.
# That's fine -- it checkpoints every epoch and resumes. To start sooner under the cap, drop
# --time to 06:00:00 (48 GPU-h) -- still ~60 epochs at ~6 min/epoch. Resume caveats verified
# in train.py for r2 checkpoints:
#   - self.iters restores (train.py:639) and is past warmup -> warmup does NOT re-fire.
#   - best_valid_metric restores via .get() (train.py:642) -> does NOT reset to 1e6.
#   - lr is force-reset to params.lr ONLY when resuming: True (train.py:646).
# Snapshot best_ckpt before any resume anyway. Raise walltime only when the account is idle.
#SBATCH --export=ALL
#SBATCH -o logs/lowres_r2_metar_%j.out
#SBATCH -e logs/lowres_r2_metar_%j.err
#
# 1/4-res Swin body, ROUND 2 -- METAR-ONLY-TRAINING variant. IDENTICAL to train_lowres_r2
# except CONFIG -> params_lowres_r2_metar.yaml (train_obs_source: metar), EXP -> lowres_r2_metar,
# and MASTER_PORT 29522. The model ingests + targets only the ~2285 METAR stations. Run
# CONCURRENTLY with lowres_r2 as a clean A/B: all-obs vs METAR-only training. Trains FROM
# SCRATCH. See config header + HANDOFF_lowres_capacity_and_lr.md.
#
# Override CONFIG/EXP to reuse this for LR/arch smokes:
#   sbatch --export=ALL,CONFIG=./config/params_lowres_r2_smoke.yaml,EXP=.../exp/lowres_r2_smoke train_lowres_r2.sbatch
#
# vs lowres_r1, r2 varies THREE coupled things (arch capacity is the primary; see the config
# header): Tier-M arch (body_dim 192, depths [4,4], heads [6,6], window 12), tighter plateau
# (patience 10, factor 0.5, warmup 2000), and heldout_metric_source: metar so best_ckpt + LR
# select on clean METAR. Every obs-dropout key, loss_channel_weights, obs_mask_seed,
# valid_max_files and batch_size is IDENTICAL to r1, and final scoring is METAR-only for
# both (heldout_eval.py --source metar), so the head-to-head bar chart stays valid.
#
# FOUR nodes x 2 H100 = 8 ranks, bs=1 -> effective batch 8 (DOUBLE r1's 4 -> lr retuned to
# 7E-4 in the config). The all-reduce is tiny (~5 M params ~= 20 MB), so cross-node comms is
# cheap even at 4 nodes. Multi-node DDP step time swings ~2x run-to-run on the shared fabric
# (memory multinode-timing-variance); this is a training run, not a benchmark, so the jitter
# only makes ms/step noisy -- the science is unaffected.
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
CONFIG="${CONFIG:-./config/params_lowres_r2_metar.yaml}"
EXP="${EXP:-/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/exp/lowres_r2_metar}"
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
export MASTER_PORT=29522  # NOT 29519 (multiscale_r1) / 29520 (lowres_r1) / 29521 (lowres_r2): avoid a collision with the concurrent r2 run
NNODES=${SLURM_NNODES:-1}

echo "==== lowres_r2_metar start $(date) nodes=${NNODES} (${SLURM_JOB_NODELIST}) master=${MASTER_ADDR} job=${SLURM_JOB_ID} config=${CONFIG} exp=${EXP} ===="
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
echo "==== lowres_r2_metar done rc=${rc} $(date) ===="
exit $rc
