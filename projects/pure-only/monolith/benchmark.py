import copy

import torch.cuda

from LJ_surrogates.surrogates.collate_data import collate_physical_property_data, calculate_ff_rmses_surrogate
from LJ_surrogates.sampling.optimize import ForceBalanceObjectiveFunction
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
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only-10-2/estimated_results'
benchmark_path = path
# path = '/home/owenmadin/Documents/python/LJ_surrogates/projects/pure-only/integrated/benchmark/estimated_results'
# benchmark_path = '/home/owenmadin/Documents/python/LJ_surrogates/projects/pure-only/integrated/benchmark/estimated_results'

smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1.0.0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only-10-2/test-set-collection.json'
device = 'cpu'

# dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
#                                           dataset_json, device)
# dataplex.plot_properties()

benchmark_dataplex = collate_physical_property_data(benchmark_path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

objective = ForceBalanceObjectiveFunction(benchmark_dataplex.multisurrogate, benchmark_dataplex.properties, benchmark_dataplex.initial_parameters,
                                          benchmark_dataplex.property_labels)
objective.flatten_parameters()
component_objectives = []
for i in range(9, benchmark_dataplex.property_measurements.shape[0]):
    component_objectives.append(objective.simulation_objective(benchmark_dataplex.property_measurements.values[i],debug=True))
# simulation_objective = objective.simulation_objective(dataplex.property_measurements.values[24])

component_objectives = np.asarray(component_objectives)

iter = np.arange(component_objectives.shape[0])

plt.plot(iter, component_objectives[:,0], label='Total Objective', color = 'k')
plt.plot(iter, component_objectives[:,1], label='Density Objective', color = 'r', ls='--')
plt.plot(iter, component_objectives[:,2], label='Hvap Objective', color = 'b', ls='--')
plt.legend()
plt.title('Objective Contributions')
plt.xlabel('Iteration')
plt.ylabel('Objective Function')
plt.savefig('objective_contributions_4.png', dpi=300)



rmses = benchmark_dataplex.calculate_ff_rmses()
# hvap_surr_rmse, density_surr_rmse = calculate_ff_rmses_surrogate(dataplex, benchmark_dataplex.parameter_values.values)
#
# abs_deviation, percent_deviation, comb_uncert, in_uncert = dataplex.calculate_surrogate_simulation_deviation(benchmark_dataplex)

# os.makedirs('benchmark',exist_ok=True)
# plt.close()
# plt.bar(['LJ Refit Mixtures','Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0'],hvap_rmse,label='Simulation RMSE',alpha = 0.5)
# plt.bar(['LJ Refit Mixtures','Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0'],hvap_surr_rmse,label='Surrogate RMSE',alpha = 0.5)
# plt.xlabel('Optimization')
# plt.ylabel('RMSE, kJ/mol')
# plt.title(r'$\Delta H_{vap}$ RMSE')
# plt.legend()
# plt.savefig(os.path.join('benchmark','hvap_rmse.png'),dpi=300)
# plt.close()
#
# plt.bar(['LJ Refit Mixtures','Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0'],density_rmse,label='Simulation RMSE',alpha = 0.5)
# plt.bar(['LJ Refit Mixtures','Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0'],density_rmse,label='Surrogate RMSE',alpha = 0.5)
# plt.xlabel('Optimization')
# plt.ylabel('RMSE, kJ/mol')
# plt.title(r'$\rho_L$ RMSE')
# plt.legend()
# plt.savefig(os.path.join('benchmark','density_rmse.png'),dpi=300)
# plt.close()
# print(rmses)
true_values = []
for property in benchmark_dataplex.properties.properties:
    true_values.append(property.value.m)

true_values = np.asarray(true_values)