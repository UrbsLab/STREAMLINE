#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=STREAMLINE
#SBATCH --mem=32G
#SBATCH -o job.o
#SBATCH -e job.e
#SBATCH --time=1-48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
source /home/${USER}/.bashrc
source activate streamline
srun python run.py -c run.cfg
