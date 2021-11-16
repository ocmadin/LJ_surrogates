#!/bin/bash

# Set the job name and wall time / memory limit
#BSUB -J server
#BSUB -W 168:00
#BSUB -M 8

# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e

#BSUB -q cpuqueue

. ~/.bashrc

# Use the right conda environment
conda activate LJ_surrogates
conda env export > conda_env.yaml

# Run the commands
python run_calculations.py -nwork 60 -ncpus 1 -ngpus 1 -dir '../modified_force_fields'
