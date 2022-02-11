from LJ_surrogates.sampling.optimize import LeastSquaresObjectiveFunction, ForceBalanceObjectiveFunction, \
    create_forcefields_from_optimized_params
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data, calculate_ff_rmses_surrogate
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
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-iterative-30'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1.0.0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/iterative-test-set-collection-initial.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
params_1 = np.asarray([0.0108,1.554,0.0783,1.956,0.1108,1.8995,0.2113,1.736,0.1658,1.759,0.2099,1.726])
# params_2 = np.load('4th_round.npy')[1]
# objective = ConstrainedGaussianObjectiveFunctionNoSurrogate(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters, 0.001)
objective = ForceBalanceObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                          dataplex.property_labels)

objective.flatten_parameters()
# objective.flat_parameters = params_2
initial_objective = objective.forward(objective.flat_parameters)

bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))

# bounds = [(0.004, 0.025), (1.0, 2), (0.05, 0.2), (1.5, 3), (0.05, 0.2), (1.5, 2.5), (0.15, 0.30), (1.2, 2.0), (0.1, 0.25), (1.2, 2.5), (0.1, 0.5), (1.2, 2.5)]
boundsrange = []
for tuple in bounds:
    boundsrange.append(tuple[1]-tuple[0])
boundsrange = np.asarray(boundsrange)

objs = []
params = []
objs_l_bfgs_b = []
params_l_bfgs_b = []

for i in range(5):
    result_l_bfgs_b = minimize(objective, objective.flat_parameters, bounds=bounds,
                               options={'maxfun': 50000, 'gtol': 1e-8, 'maxls': 50})
    before = time.time()
    result_de = differential_evolution(objective, bounds, popsize=20, tol=0.001, recombination=0.9)
    after = time.time()
    print(f'DE Time: {after - before} seconds')
    objs.append(result_de.fun)
    params.append(result_de.x)
    objs_l_bfgs_b.append(result_l_bfgs_b.fun)
    params_l_bfgs_b.append(result_l_bfgs_b.x)

simulation_opt = np.asarray(
    [0.008766206, 1.46527, 0.080329, 1.998187, 0.0993459, 1.9809416, 0.20698197, 1.7208416, 0.16197438,
     1.7737039, 0.2106341, 1.71455])

simulation_objective = objective.forward(simulation_opt)

params_to_simulate = [simulation_opt, params[np.argmin(objs)], params_l_bfgs_b[np.argmin(objs_l_bfgs_b)]]

objs_to_simulate = [simulation_objective,min(objs), min(objs_l_bfgs_b)]

create_forcefields_from_optimized_params(params_to_simulate, objective.flat_parameter_names, 'openff-1.0.0.offxml')

params_to_simulate.append(objective.flat_parameters)

objs_to_simulate.append(objective.forward(objective.flat_parameters))

params_to_simulate = np.asarray(params_to_simulate)


new_bounds = np.asarray([(params_to_simulate[1][i]-0.1*boundsrange[i],params_to_simulate[1][i] + 0.1*boundsrange[i]) for i,param in enumerate(params_to_simulate[1])])

np.save('new_bounds.npy',new_bounds)
np.save('new_parameters.npy',params_to_simulate[1])



all_to_simulate = np.load('../monolith/all_monolith_params.npy')
hvap_rmse, density_rmse = calculate_ff_rmses_surrogate(dataplex, all_to_simulate)

results = []
rmses = []
for i in range(all_to_simulate.shape[1]):
    results.append(objective.forward(all_to_simulate[i]))
