import torch.cuda

from LJ_surrogates.surrogates.collate_data import collate_physical_property_data, calculate_ff_rmses_surrogate
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
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-integrated-test-full'
benchmark_path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-integrated-test-full'
# path = '/home/owenmadin/Documents/python/LJ_surrogates/projects/pure-only-integrated-test-full'
# benchmark_path = '/home/owenmadin/Documents/python/LJ_surrogates/projects/pure-only-integrated-test-full'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/iterative-test-set-collection-initial.json'
# dataset_json = '/home/owenmadin/Documents/python/LJ_surrogates/projects/pure-only/integrated/benchmark/optimized_ffs/1/test-set-collection.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
# dataplex.plot_properties()

benchmark_dataplex = collate_physical_property_data(benchmark_path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

hvap_rmse, density_rmse, hmix_rmse, binary_density_rmse = benchmark_dataplex.calculate_ff_rmses()
# hvap_surr_rmse, density_surr_rmse = calculate_ff_rmses_surrogate(dataplex, benchmark_dataplex.parameter_values.values)

abs_deviation, percent_deviation, comb_uncert, in_uncert = dataplex.calculate_surrogate_simulation_deviation(benchmark_dataplex)

os.makedirs('benchmark',exist_ok=True)
plt.close()
# plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],hvap_rmse,label='Simulation RMSE',alpha = 0.5)
# plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],hvap_surr_rmse,label='Surrogate RMSE',alpha = 0.5)
# plt.xlabel('Optimization')
# plt.ylabel('RMSE, kJ/mol')
# plt.title(r'$\Delta H_{vap}$ RMSE')
# plt.legend()
# plt.savefig(os.path.join('benchmark','hvap_rmse.png'),dpi=300)
# plt.close()
#
# plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],density_rmse,label='Simulation RMSE',alpha = 0.5)
# plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],density_rmse,label='Surrogate RMSE',alpha = 0.5)
# plt.xlabel('Optimization')
# plt.ylabel('RMSE, kJ/mol')
# plt.title(r'$\rho_L$ RMSE')
# plt.legend()
# plt.savefig(os.path.join('benchmark','density_rmse.png'),dpi=300)
# plt.close()

true_values = []
for property in benchmark_dataplex.properties.properties:
    true_values.append(property.value.m)

true_values = np.asarray(true_values)

# print(true_values)
#
# print(benchmark_dataplex.property_measurements.values[9])
# print(benchmark_dataplex.property_measurements.values[24])
# print(benchmark_dataplex.property_measurements.values[20])


mfo = np.asarray([density_rmse[0],hvap_rmse[0],binary_density_rmse[0],hmix_rmse[0]])
off100 = np.asarray([0.03,9.92,0.025,0.61])
sim_opt = np.asarray([0.018, 7.47, 0.018, 0.44])

all_benchmarks = np.stack((off100,sim_opt,mfo)).T

fig,ax = plt.subplots(1,4, figsize=(16,5))
colors = ['blue','orange','green']
labels = ['OpenFF 1.0.0', 'Simulation \n only', 'Multi-Fidelity']
props = [r'$\rho_l$',r'$\Delta H_{vap}$', r'$\rho_l(x)$', r'$\Delta H_{mix}(x)$']
units = ['g/mL', 'kJ/mol', 'g/mL', 'kJ/mol']
for i in range(len(props)):
    ax[i].bar(labels,all_benchmarks[i], color=colors)
    ax[i].set_ylabel(r'RMSE, '+units[i], fontsize=16)
    ax[i].set_title(props[i], fontsize=16)
    # ax[i].tick_params('x',labelrotation=45)
fig.suptitle('Optimization Benchmarking', fontsize=20)
fig.tight_layout()
fig.savefig('rmse_comparison.png', dpi=300)
fig.show()