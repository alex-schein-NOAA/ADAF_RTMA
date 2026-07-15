#!/bin/bash
#SBATCH --job-name=adaf_preflight_lowres_r2
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --export=ALL
#SBATCH -o logs/preflight_lowres_r2_%j.out
#SBATCH -e logs/preflight_lowres_r2_%j.err
#
# r2 Tier-M memory + throughput preflight (HANDOFF_lowres_capacity_and_lr.md S5). Real
# fwd+bwd at 1356x2294 under production precision (bf16 + channels_last).
# GO/NO-GO: peak @ bs=2 must be < ~60 GiB with headroom (lowres_r1 was 28.0; the flat model
# 53.2; the pyramid 79.2 with no headroom -> its failure mode). r2 is ~4.99 M params (r1:
# 1.98 M). Compared against r1 and the flat default so the delta is explicit.
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
CLONE_ENV=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate "$CLONE_ENV"
export PATH="${CLONE_ENV}/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p logs

echo "==== preflight_lowres_r2 start $(date) host=$(hostname) job=${SLURM_JOB_ID} ===="
python preflight_memory.py \
    --configs ./config/params_lowres.yaml ./config/params_lowres_r2.yaml \
    --batch_sizes 1 2 4
echo "==== preflight_lowres_r2 done rc=$? $(date) ===="
