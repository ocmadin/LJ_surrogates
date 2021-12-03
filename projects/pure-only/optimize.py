from LJ_surrogates.sampling.optimize import ConstrainedGaussianObjectiveFunction, create_forcefields_from_optimized_params
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
import torch
import gc
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import numpy as np
import pandas
import textwrap
import seaborn
import os
from LJ_surrogates.plotting.plotting import plot_triangle

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-1-3-0-100-samples'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = 'pure-only-collection.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

objective = ConstrainedGaussianObjectiveFunction(dataplex.surrogates, dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                                 0.1)
objective.flatten_parameters()
bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))

objs = []
params = []
objs_l_bfgs_b = []
params_l_bfgs_b = []
for i in range(5):
    result_l_bfgs_b = minimize(objective,objective.flat_parameters, bounds=bounds, method='L-BFGS-B')
    result_de = differential_evolution(objective, bounds)
    objs.append(result_de.fun)
    params.append(result_de.x)
    objs_l_bfgs_b.append(result_l_bfgs_b.fun)
    params_l_bfgs_b.append(result_l_bfgs_b.x)
