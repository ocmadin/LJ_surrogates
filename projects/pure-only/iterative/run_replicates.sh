#!/bin/bash

conda activate LJ_surrogates

number_of_replicates=10

python prepare_simulations.py -n $number_of_replicates -o 'modified_force_fields' -f 'openff-1.0.0.offxml' -d 'pure-only.csv' -r 'new_bounds.npy' -a True

cd run_server
bsub < server-submit.sh
cd ..
