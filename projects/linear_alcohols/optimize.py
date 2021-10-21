from LJ_surrogates.sampling.optimize import UnconstrainedGaussianObjectiveFunction, ConstrainedGaussianObjectiveFunction
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
import torch
import gc
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import numpy as np
from LJ_surrogates.plotting.plotting import plot_triangle

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/alcohol_alkane/linear_alcohols_2nd_refinement'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', '[#1:1]-[#8]', '[#6X4:1]',
                          '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = 'linear_alcohols_refined_new.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

objective = ConstrainedGaussianObjectiveFunction(dataplex.surrogates, dataplex.properties, dataplex.initial_parameters,
                                                 0.1)
objective.flatten_parameters()
bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))

objs = []
params = []
for i in range(1):
    result = differential_evolution(objective, bounds)
    objs.append(result.fun)
    params.append(result.x)

samples = np.load('result_100k_10_21/params.npy')[::100]
plot_triangle(samples, objective, None, params[0], boundaries=False, maxima=True)
