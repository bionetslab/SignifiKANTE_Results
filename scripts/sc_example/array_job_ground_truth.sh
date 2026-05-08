#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --array=1-30%30
#SBATCH --output=logs_gt/slurm_%A_%a.out
#SBATCH --error=logs_gt/slurm_%A_%a.err

unset SLURM_EXPORT_ENV

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for Slurm version >22.05: cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# activate conda env
module purge
source /home/hpc/iwbn/iwbn107h/miniforge3/etc/profile.d/conda.sh
conda activate sk

# change to subdirectory
cd "$WORK/projects/SignifiKANTE_Results/scripts/sc_example"

# get config files
CONFIG_DIR="./configs_ground_truth"
mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" | sort)

# select config file
CONFIG_FILE=${CONFIG_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}

# run
echo "Running classical FDR for: $CONFIG_FILE"
srun python ./computation_ground_truth.py -f "$CONFIG_FILE"





















