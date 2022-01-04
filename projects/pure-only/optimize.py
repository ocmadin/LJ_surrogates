from LJ_surrogates.sampling.optimize import ConstrainedGaussianObjectiveFunction, create_forcefields_from_optimized_params
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
import torch
import gc
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, least_squares
import numpy as np
import pandas
import textwrap
import seaborn
import os
from LJ_surrogates.plotting.plotting import plot_triangle
import time

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-100-1-0-0'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/test-set-collection-100-1-0-0.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

objective = ConstrainedGaussianObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                                 0.1)
objective.flatten_parameters()
bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))

objs = []
params = []
objs_lm = []
params_lm = []

for i in range(5):
    result_l_bfgs_b = minimize(objective,objective.flat_parameters, bounds=bounds, method='L-BFGS-B')
    before = time.time()
    result_de = differential_evolution(objective, bounds)
    after = time.time()
    print(f'DE Time: {after-before} seconds')
    objs.append(result_de.fun)
    params.append(result_de.x)
    objs_lm.append(result_l_bfgs_b.fun)
    params_lm.append(result_l_bfgs_b.x)

simulation_opt = np.asarray([0.009, 1.465, 0.08 , 1.998, 0.099, 1.981, 0.207, 1.721, 0.162,
       1.774, 0.211, 1.715])

params_to_simulate = [simulation_opt,params[np.argmin(objs)], params_lm[np.argmin(objs_lm)]]

create_forcefields_from_optimized_params(params_to_simulate,objective.flat_parameter_names,'openff-1.0.0.offxml')

