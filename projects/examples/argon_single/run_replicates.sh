#!/bin/bash

conda activate LJ_surrogates

number_of_replicates=20

python prepare_simulations.py -n $number_of_replicates -o 'modified_force_fields' -f 'openff-1-3-0-argon.offxml' -d 'argon_single.csv'
cd run_server
bsub < server-submit.sh
cd ..
