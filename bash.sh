#!/bin/bash
# Set Job Requirements
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --partition=genoa
#SBATCH --out=slurm/slurm-%A.out
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=b.t.markhorst@student.vu.nl

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Gurobi/10.0.1-GCCcore-11.3.0
export GRB_LICENSE_FILE="gurobi-2.lic"

python FodstadExample.py