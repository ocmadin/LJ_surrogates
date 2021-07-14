#!/bin/bash

conda activate LJ_surrogates

number_of_replicates=2

for i in $(seq 1 $number_of_replicates)
do
   cd modified_force_fields/$i
   bsub < submit.sh
   sleep 0.5
   cd ..
done

