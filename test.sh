#!/bin/bash
#BSUB -q i2c2_normal
#BSUB -J hw_test
#BSUB -o log.o
#BSUB -e log.e
python -c "print('Hello World')"