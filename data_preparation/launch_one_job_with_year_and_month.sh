#!/bin/bash

# Check if both year and month arguments were passed
if [ "$#" -ne 3 ]; then
    echo "Error: Missing arguments."
    echo "Usage:   $0 <year> <month> <obs_source>"
    echo "Example: $0 2021 05 ''combined'' "
    exit 1
fi

# Assign inputs to descriptive variables
YEAR=$1
MONTH=$2
OBS_SOURCE=$3

PYTHON_SCRIPT="sample_generate.py"

# Dynamically route the save directory based on the user-inputted year
case "$YEAR" in
    2021) save_dir="/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/train_data" ;;
    2022) save_dir="/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/valid_data" ;;
    2023) save_dir="/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/test_data" ;;
    *)
        echo "Error: Unsupported year '$YEAR'. Supported years are 2021, 2022, or 2023."
        exit 1
        ;;
esac

# Dynamically calculate the last day of the user-inputted month
case "$MONTH" in
    01|03|05|07|08|10|12) last_day="31" ;;
    04|06|09|11)          last_day="30" ;;
    02)                   last_day="28" ;; # 2021-2023 are all non-leap years
    *)
        echo "Error: Invalid month '$MONTH'. Please use MM format (01 through 12)."
        exit 1
        ;;
esac

# Construct time variables
start_time="${YEAR}-${MONTH}-01_03"  #00 if doing anything other than jan 2021, 01 if doing jan 2021
end_time="${YEAR}-${MONTH}-${last_day}_23"

# Submit directly to SLURM
sbatch <<EOT
#!/bin/bash
#SBATCH -A wrfruc
#SBATCH -p u1-compute
#SBATCH --job-name=analysis_${YEAR}_${MONTH}
#SBATCH --output=logs/analysis_${YEAR}_${MONTH}_%J.out
#SBATCH --error=logs/analysis_${YEAR}_${MONTH}_%J.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=12G

export HDF5_USE_FILE_LOCKING=FALSE

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh

unset PYTHONPATH

echo "Starting job for $start_time to $end_time"

/scratch3/BMC/wrfruc/aschein/miniconda/envs/ADAF_environment/bin/python -u "$PYTHON_SCRIPT" \\
    --starting_analysis_time "$start_time" \\
    --ending_analysis_time "$end_time" \\
    --obs_source "$OBS_SOURCE" \\
    --save_directory "$save_dir"
EOT
