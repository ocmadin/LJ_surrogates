import torch.cuda

from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
from LJ_surrogates.parameter_modification import vary_parameters_lhc, create_evaluation_grid
import time
from LJ_surrogates.sampling.likelihood import likelihood_function
import torch
import pickle
import pandas
import seaborn
import matplotlib.pyplot as plt
import gc
import numpy as np
import os
import textwrap
from LJ_surrogates.plotting.plotting import plot_triangle

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-100-1-0-0'
benchmark_path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-benchmark-100-1-0-0'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/test-set-collection-100-1-0-0.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
dataplex.plot_properties()

benchmark_dataplex = collate_physical_property_data(benchmark_path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

hvap_rmse, density_rmse = dataplex.calculate_ff_rmses()

abs_deviation, percent_deviation, comb_uncert, in_uncert = dataplex.calculate_surrogate_simulation_deviation(benchmark_dataplex)
