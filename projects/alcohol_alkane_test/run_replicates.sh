#!/bin/bash

conda activate LJ_surrogates

number_of_replicates=4

python prepare_simulations.py -n $number_of_replicates -o 'modified_force_fields' -f 'openff-1-3-0.offxml' -d 'alcohol_alkane_datapoints.csv'

cd run_server
bsub < submit.sh
cd ..

for i in $(seq 1 $number_of_replicates)
do
   cd modified_force_fields/$i
   bsub < submit.sh
   sleep 0.5
   cd ../..
done

