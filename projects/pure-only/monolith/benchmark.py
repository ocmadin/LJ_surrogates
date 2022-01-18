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
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-100-1-0-0'
benchmark_path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-new-benchmark-100-1-0-0'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/test-set-collection-100-1-0-0.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
dataplex.plot_properties()

benchmark_dataplex = collate_physical_property_data(benchmark_path, smirks_types_to_change, forcefield,
                                          dataset_json, device)

hvap_rmse, density_rmse = benchmark_dataplex.calculate_ff_rmses()
hvap_surr_rmse, density_surr_rmse = calculate_ff_rmses_surrogate(dataplex, benchmark_dataplex.parameter_values.values)

abs_deviation, percent_deviation, comb_uncert, in_uncert = dataplex.calculate_surrogate_simulation_deviation(benchmark_dataplex)

os.makedirs('benchmark',exist_ok=True)
plt.close()
plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],hvap_rmse,label='Simulation RMSE',alpha = 0.5)
plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],hvap_surr_rmse,label='Surrogate RMSE',alpha = 0.5)
plt.xlabel('Optimization')
plt.ylabel('RMSE, kJ/mol')
plt.title(r'$\Delta H_{vap}$ RMSE')
plt.legend()
plt.savefig(os.path.join('benchmark','hvap_rmse.png'),dpi=300)
plt.close()

plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],density_rmse,label='Simulation RMSE',alpha = 0.5)
plt.bar(['Surrogate DE','Surrogate L-BFGS-B','OpenFF 1.0.0','LJ Refit Mixtures'],density_rmse,label='Surrogate RMSE',alpha = 0.5)
plt.xlabel('Optimization')
plt.ylabel('RMSE, kJ/mol')
plt.title(r'$\rho_L$ RMSE')
plt.legend()
plt.savefig(os.path.join('benchmark','density_rmse.png'),dpi=300)
plt.close()