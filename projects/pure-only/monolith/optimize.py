from LJ_surrogates.sampling.optimize import LeastSquaresObjectiveFunction, ForceBalanceObjectiveFunction, \
    create_forcefields_from_optimized_params, ConstrainedGaussianObjectiveFunction
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
from gpytorch.constraints import GreaterThan
import time

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-pooled'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1.0.0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/test-set-collection.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device, constraint=None)

# objective = ConstrainedGaussianObjectiveFunctionNoSurrogate(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters, 0.001)
# objective = ConstrainedGaussianObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
#                                           0.1)
objective = ForceBalanceObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                          dataplex.property_labels)
objective.flatten_parameters()

initial_objective = objective.forward(objective.flat_parameters)
# jacobian = objective.forward_jac(objective.flat_parameters)
simulation_opt = np.asarray(
    [0.008766206, 1.46527, 0.080329, 1.998187, 0.0993459, 1.9809416, 0.20698197, 1.7208416, 0.16197438,
     1.7737039, 0.2106341, 1.71455])
DE_values = np.asarray([0.0079285 , 1.46571923, 0.09187845, 2.00240791, 0.0985694 ,
       1.98904371, 0.20147424, 1.7404901 , 0.16501403, 1.76432761,
       0.20605592, 1.70430578])
bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))
bounds = np.asarray(bounds)
bounds_increment = 1.1
bounds[:, 0] /= (bounds_increment ** (1))
bounds[:, 1] *= (bounds_increment ** (1))

# bounds = [(0.004, 0.025), (1.0, 2), (0.05, 0.2), (1.5, 3), (0.05, 0.2), (1.5, 2.5), (0.15, 0.30), (1.2, 2.0), (0.1, 0.25), (1.2, 2.5), (0.1, 0.5), (1.2, 2.5)]

boundsrange = []
for tuple in bounds:
    boundsrange.append(tuple[1]-tuple[0])
boundsrange = np.asarray(boundsrange)

objs = []
params = []
objs_lbfgsb = []
params_lbfgsb = []
objs_bfgs = []
params_bfgs = []
objs_ncg = []
params_ncg = []

def callbackF(objective):
    print(objective)

for i in range(1):
    # result_l_bfgs_b = minimize(objective,x0=objective.flat_parameters, jac=objective.forward_jac, bounds=bounds, method='L-BFGS-B')
    before = time.time()
    result_de = differential_evolution(objective, bounds)
    after = time.time()
    print(f'DE Time: {after - before} seconds')
    objs.append(result_de.fun)
    params.append(result_de.x)
    # objs_lbfgsb.append(result_l_bfgs_b.fun)
    # params_lbfgsb.append(result_l_bfgs_b.x)
    # result_ncg = minimize(objective, x0=objective.flat_parameters,method='Newton-CG', jac=objective.forward_jac)
    # objs_ncg.append(result_ncg.fun)
    # params_ncg.append(result_ncg.x)
    # # result_bfgs = minimize(objective,objective.flat_parameters,method='BFGS', jac=objective.forward_jac)
    # objs_bfgs.append(result_bfgs.fun)
    # params_bfgs.append(result_bfgs.x)

lengthscales = []

for model in dataplex.multisurrogate.models:
    lengthscales.append(model.covar_module.base_kernel.lengthscale)

lengthscales = torch.stack(lengthscales).squeeze(1).detach().numpy()


objectives = []
for i in range(dataplex.property_measurements.values.shape[0]):
    objectives.append(objective.simulation_objective(dataplex.property_measurements.values[i]))


new_df = dataplex.parameter_values
new_df['objective'] = np.asarray(objectives)
new_df.to_csv('../../mixture-only/df_10.csv')
experimental_values = []
# cl_indices = []
# br_indices = []
# clbr_indices = []
for i,property in enumerate(dataplex.properties.properties):
    experimental_values.append(property.value.m)
    # if 'Cl' in property.substance.identifier and 'Br' not in property.substance.identifier:
    #     cl_indices.append(i)
    # if 'Br' in property.substance.identifier and 'Cl' not in property.substance.identifier:
    #     br_indices.append(i)
    # if 'Cl' in property.substance.identifier and 'Br'  in property.substance.identifier:
    #     clbr_indices.append(i)

experimental_values = np.asarray(experimental_values)

# cl_experimental_values = experimental_values[cl_indices]
# cl_experimental_values = np.append(cl_experimental_values, [1.487, 1.594])
# cl_openff_1_0_0_values = np.append(dataplex.property_measurements.values[0][cl_indices], [1.4489, 1.58979])
# cl_openff_2_0_0_values = np.append(dataplex.property_measurements.values[1][cl_indices], [1.585, 1.742])
#
# br_experimental_values = np.append(experimental_values[br_indices], 2.878)
# br_openff_1_0_0_values = np.append(dataplex.property_measurements.values[0][br_indices], 2.247)
# br_openff_2_0_0_values = np.append(dataplex.property_measurements.values[1][br_indices], 2.779)
#
# clbr_experimental_values = experimental_values[clbr_indices]
# clbr_openff_1_0_0_values = dataplex.property_measurements.values[0][clbr_indices]
# clbr_openff_2_0_0_values = dataplex.property_measurements.values[1][clbr_indices]
#
# surrogate_values = dataplex.multisurrogate.posterior(torch.tensor(params[0]).unsqueeze(0)).mean.detach().numpy().squeeze()
#
# plt.scatter(cl_experimental_values, cl_openff_1_0_0_values, color='b', marker='x', label='Cl Only, OpenFF 1.0.0')
# plt.scatter(cl_experimental_values, cl_openff_2_0_0_values, color='orange', marker='x', label='Cl Only, OpenFF 2.0.0')
#
# plt.scatter(br_experimental_values, br_openff_1_0_0_values, color='b', marker='o', facecolor='none', label='Br Only, OpenFF 1.0.0')
# plt.scatter(br_experimental_values, br_openff_2_0_0_values, color='orange', marker='o', facecolor='none', label='Br Only, OpenFF 2.0.0')
#
# plt.scatter(clbr_experimental_values, clbr_openff_1_0_0_values, color='b', marker='s', facecolor='none', label='Cl + Br, OpenFF 1.0.0')
# plt.scatter(clbr_experimental_values, clbr_openff_2_0_0_values, color='orange', marker='s', facecolor='none', label='Cl + Br, OpenFF 2.0.0')
#
# plt.plot([0.75,3.0],[0.75,3.0], color='k')
# plt.text(1.45, 1.55, r'CHCl$_3$', horizontalalignment='right')
# plt.text(1.55, 1.75, r'CCl$_4$', horizontalalignment='right')
# plt.text(1.4, 1.45, r'C$_2$H$_3$Cl$_3$', horizontalalignment='right')
# plt.xlim([0.75,3.0])
# plt.ylim([0.75,3.0])
# plt.xlabel('Experimental Density, g/ml')
# plt.ylabel('Simulation Density, g/ml')
# plt.title('Density of Halogenated Alkanes')
# plt.legend()
# plt.tight_layout()
# plt.show()

# compare_df = dataplex.property_measurements.loc[dataplex.property_measurements.shape[0]]