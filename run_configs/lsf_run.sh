#!/bin/bash
#BSUB -p i2c2_normal
#BSUB -J STREAMLINE
#BSUB -R -"rusage[mem=1G]"
#BSUB -M 1G
#BSUB -o job.o
#BSUB -e job.e
#BSUB --time=48:00:00
python run.py -c upenn.cfg
