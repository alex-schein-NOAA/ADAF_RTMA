#!/bin/bash
#SBATCH --job-name=adaf_preflight_lowres
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
#SBATCH -o logs/preflight_lowres_%j.out
#SBATCH -e logs/preflight_lowres_%j.err
#
# P2 (HANDOFF_lowres_arch.md S5): peak memory + ms/sample for the 1/4-res body against the
# flat model, real fwd+bwd at 1356x2294 under production precision (bf16 + channels_last).
# GO/NO-GO: peak at bs=2 must be < 60 GiB (flat is 53.2; the pyramid was 79.2 with no
# headroom, which is what forced its design into the failure mode). If bs=4 fits under
# ~80 GiB, note it -- but using it means retuning lr and voiding the clean A/B vs dropout_r1.
set -uo pipefail

REPO=/scratch3/BMC/wrfruc/Micah.Craine/ADAF_RTMA/ADAF_RTMA
CLONE_ENV=/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate "$CLONE_ENV"
export PATH="${CLONE_ENV}/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p logs

echo "==== preflight_lowres start $(date) host=$(hostname) job=${SLURM_JOB_ID} ===="
python preflight_memory.py \
    --configs ./config/params_default.yaml ./config/params_lowres.yaml \
    --batch_sizes 1 2 4
echo "==== preflight_lowres done rc=$? $(date) ===="
