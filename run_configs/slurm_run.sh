#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=STREAMLINE
#SBATCH --mem=1G
#SBATCH -o job.o
#SBATCH -e job.e
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
srun python run.py -c cedars.cfg
