from LJ_surrogates.parameter_modification import vary_parameters_lhc
import os
import shutil
forcefield = 'openff-1-3-0.offxml'
num_samples = 2
output_directory = 'modified_force_fields'
data_filepath = 'alcohol_alkane_datapoints_test.csv'

vary_parameters_lhc(forcefield,num_samples,output_directory)
