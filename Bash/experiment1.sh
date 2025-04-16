#!/bin/bash
# Set Job Requirements
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH -n 16
#SBATCH --partition=genoa
#SBATCH --out=slurm/slurm-%A_%a.out
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=b.t.markhorst@student.vu.nl

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Gurobi/10.0.1-GCCcore-11.3.0

export GRB_LICENSE_FILE="gurobi-2.lic"

python FodstadExample.py --input_file Data/OneScenario/SanityCheckBig.xlsx --output_file Results/Experiment1/base_case --equality_constraint 1 --crossover 1
