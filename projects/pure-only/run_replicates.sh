#!/bin/bash

conda activate lj-surrogates-mcmc

number_of_replicates=40

python prepare_simulations.py -n $number_of_replicates -o 'modified_force_fields' -f 'openff-1-3-0.offxml' -d 'pure-alcohols.csv'

cd run_server
#bsub < server-submit.sh
cd ..


