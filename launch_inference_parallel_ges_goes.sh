#!/bin/bash
#SBATCH --account=gpu-wizard
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --job-name=ADAF_RTMA_inference
#SBATCH --output=inference_runs/%j/inference_%j.out
#SBATCH --error=inference_runs/%j/inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${ADAF_RTMA_DIR:-/scratch3/BMC/wrfruc/aschein/ADAF_RTMA}"
CONDA_ROOT="${ADAF_CONDA_ROOT:-/scratch3/BMC/wrfruc/aschein/miniconda}"
start_time=()
end_time=()

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: sbatch [Slurm options] launch_inference_parallel_ges_goes.sh [--start_time HOUR] [--end_time HOUR]' \
        '' \
        'Edit inference_command near the bottom for all options other than time bounds.' \
        'Uncomment optional flags in that same block to enable them.' \
        'Edit the ending of --output_dir for the model or experiment name.' \
        'Each job uses one node and one GPU.' \
        '' \
        '  --start_time YYYY-MM-DD_HH    Inclusive lower limit; default: no limit' \
        '  --end_time YYYY-MM-DD_HH      Inclusive upper limit; default: no limit' \
        '' \
        'Example:' \
        '  sbatch launch_inference_parallel_ges_goes.sh --start_time 2023-01-02_02 --end_time 2023-01-02_07' \
        'Relative inference paths are resolved from ADAF_RTMA, not the submission directory.' \
        'Run bash launch_inference_parallel_ges_goes.sh --help to show this help without submitting.'
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_time|--end_time)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                printf '%s requires a value.\n' "$1" >&2
                exit 1
            fi
            if [[ "$1" == "--start_time" ]]; then
                start_time=(--start_time "$2")
            else
                end_time=(--end_time "$2")
            fi
            shift 2
            ;;
        *)
            printf 'Only --start_time and --end_time are command-line options; edit the script for %s.\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Submit this script using sbatch; use --help for usage.' >&2
    exit 1
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]]; then
    printf '%s\n' 'This launcher supports one node per job; submit multiple jobs for parallel inference.' >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# module load python
unset PYTHONPATH
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda activate ADAF_environment; then
    printf '%s\n' 'Failed to activate ADAF_environment.' >&2
    exit 1
fi
set -u
cd "${PROJECT_DIR}"

job_start_time=$(date +%s)
printf 'Starting inference job %s on %s at %s\n' "${SLURM_JOB_ID}" "$(hostname)" "$(date)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"

#### EDIT THIS TO LOAD A DIFFERENT MODEL
### EDIT --output_dir BELOW TO CHANGE THE SAVE DIRECTORY
CKPT_DIR="lowres_EXP_22743744_resume_23412634" #lowres_EXP_22743744_resume_23081675

inference_command=(
    "${CONDA_ROOT}/envs/ADAF_environment/bin/python" -u
    "${PROJECT_DIR}/inference_parallel_ges_goes.py"
    --config_filepath "${PROJECT_DIR}/config/params_lowres_ges_goes.yaml"
    --checkpoint_path "/scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/${CKPT_DIR}/best_ckpt.tar"
    --stats_path "${PROJECT_DIR}/data_preparation_ges/stats_ges.csv"
    --input_dir "/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes/test_data"
    --output_dir "${INFERENCE_OUTPUT_DIR:-/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/inference_output}/${CKPT_DIR}" #_hold_out_obs"
    --batch_size 8
    --num_workers 3
    --prefetch_factor 2
    --hold_out_obs False
    --hold_out_obs_ratio 0.0 #0.1
    --overwrite
    # --deny_goes_channels goes_channel_02 goes_channel_07
    # --deny_goes_hours 0 2
    "${start_time[@]}"
    "${end_time[@]}"
)

printf 'Inference command:\n '
for arg in "${inference_command[@]}"; do
    [[ "${arg}" == --* ]] && printf ' \\\n   '
    printf ' %q' "${arg}"
done
printf '\n'

srun --nodes=1 --ntasks=1 --mpi=none "${inference_command[@]}" #--gres=gpu:1

printf 'Inference job %s complete; elapsed=%s seconds\n' \
    "${SLURM_JOB_ID}" "$(( $(date +%s) - job_start_time ))"