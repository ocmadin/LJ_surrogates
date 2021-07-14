from LJ_surrogates.run_simulations import estimate_forcefield_properties
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
import pandas
import os

forcefield_file = 'openff-1-3-0-LHS-modified.offxml'
data_filepath = 'alcohol_alkane_datapoints_test.csv'
data_csv = pandas.read_csv(data_filepath)
data_csv['Id'] = data_csv['Id'].astype('string')
data_set = PhysicalPropertyDataSet.from_pandas(data_csv)
output_directory = 'modified_force_fields'

for folder in os.listdir(output_directory):
    force_field = ForceField(os.path.join(output_directory,folder,forcefield_file))
    estimate_forcefield_properties(data_set,force_field,os.path.join(output_directory,folder))
    print(f'Estimated Properties for Perturbation {folder}')