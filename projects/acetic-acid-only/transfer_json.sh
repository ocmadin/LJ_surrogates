#!/bin/bash
number_of_replicates = 100
filepath = 'acetic-acid-only.json'


for i in {1..100}
do
    cp acetic-acid-only.json optimized_ffs/$i/test-set-collection.json
done
