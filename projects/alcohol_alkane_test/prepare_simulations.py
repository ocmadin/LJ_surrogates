from LJ_surrogates.LJ_surrogates.parameter_modification import vary_parameters_lhc
import os
import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
import shutil
forcefield = 'openff-1-3-0.offxml'
num_samples = 4
output_directory = 'modified_force_fields'
data_filepath = 'alcohol_alkane_datapoints_test.csv'
data_csv = pandas.read_csv(data_filepath)
data_csv['Id'] = data_csv['Id'].astype('string')
data_set = PhysicalPropertyDataSet.from_pandas(data_csv)
data_set.json('test-set-collection.json')


vary_parameters_lhc(forcefield,num_samples,output_directory)

for folder in os.listdir(output_directory):
    shutil.copy2('test-set-collection.json',os.path.join(output_directory,folder))
    shutil.copy2('server-config.json', os.path.join(output_directory, folder))
    shutil.copy2('estimation-options.json', os.path.join(output_directory, folder))
    shutil.copy2('submit.sh', os.path.join(output_directory, folder))
    shutil.copy2('benchmark.json', os.path.join(output_directory,folder))
