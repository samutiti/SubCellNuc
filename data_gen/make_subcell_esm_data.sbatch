#!/bin/bash
#SBATCH --job-name=make_subcell_esm_data   # Name of your job
#SBATCH --output=make_subcell_esm_data_%j.out    # Standard output file
#SBATCH --error=make_subcell_esm_data_%j.err     # Standard error file
#SBATCH --time=06:00:00        # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks (processes)
#SBATCH --cpus-per-task=4      # Number of CPU cores per task
#SBATCH --mem=32G               # Memory per node/task

# Your commands go here
module load python/3.12
source ../../env/bin/activate
python make_subcell_esm_data_2.py