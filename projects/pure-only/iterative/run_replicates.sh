#!/bin/bash

conda activate LJ_surrogates

number_of_replicates=10

python prepare_simulations.py -n $number_of_replicates -o 'modified_force_fields' -f 'iterative-ff-1.offxml' -d 'pure-only.csv'

cd run_server
bsub < server-submit.sh
cd ..


