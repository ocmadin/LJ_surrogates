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
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/alcohol_alkane/linear_alcohols_optimized'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', '[#1:1]-[#8]', '[#6X4:1]',
                          '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = 'linear_alcohols_refined_new.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
dataplex.plot_properties()

hvap_rmse, density_rmse = dataplex.calculate_ff_rmses()
