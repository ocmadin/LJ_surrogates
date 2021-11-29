#!/bin/bash

conda activate LJ_surrogates

number_of_replicates=50

python prepare_simulations.py -n $number_of_replicates -o 'modified_force_fields' -f 'openff-1-3-0.offxml' -d 'pure-only.csv'

cd run_server
bsub < server-submit.sh
cd ..


