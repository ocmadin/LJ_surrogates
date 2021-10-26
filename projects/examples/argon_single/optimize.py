import torch.cuda
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
from LJ_surrogates.parameter_modification import vary_parameters_lhc, create_evaluation_grid
from LJ_surrogates.sampling.likelihood import likelihood_function
import torch
import matplotlib.pyplot as plt
import gc
import numpy as np
import os
from LJ_surrogates.sampling.optimize import UnconstrainedGaussianObjectiveFunction, ConstrainedGaussianObjectiveFunction
from scipy.optimize import differential_evolution, minimize, brute

gc.collect()
torch.cuda.empty_cache()
path = '../../../data/argon-single-50-small'
smirks_types_to_change = ['[#18:1]']
forcefield = 'openff-1-3-0-argon.offxml'
dataset_json = 'argon_single.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
# dataplex.initial_parameters['[#18:1]'][0]._value = 0.35
# dataplex.initial_parameters['[#18:1]'][1]._value = 1.5

objective = ConstrainedGaussianObjectiveFunction(dataplex.surrogates, dataplex.multisurrogate, dataplex.properties,
                                                 dataplex.initial_parameters, 0.1)
objective.flatten_parameters()
bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))
objs = []
params = []
objs_gd = []
params_gd = []
for i in range(10):
    result = differential_evolution(objective, bounds)
    objs.append(result.fun)
    params.append(result.x)
result_brute = brute(objective, bounds, Ns=100, full_output=True, finish=None)
result_gd = minimize(objective, np.array([0.2, 1.5]), bounds=bounds, method='L-BFGS-B')
result_gd_ub = minimize(objective, np.array([0.2,1.5]), method='BFGS')
objs_gd.append(result_gd.fun)
params_gd.append(result_gd.x)
params = np.asarray(params)
params_gd = np.asarray(params_gd).T
params_gd_ub = result_gd_ub.x
grid = create_evaluation_grid(forcefield, smirks_types_to_change, np.array([0.25, 1.75]))
likelihood = likelihood_function(dataplex, device)


def grid_to_surrogate_2D(grid, surrogate):
    value_grid = np.empty((grid[0].shape[0], grid[0].shape[1]))
    uncertainty_grid = np.empty((grid[0].shape[0], grid[0].shape[1]))

    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            # val = surrogate(
            #     torch.tensor(np.expand_dims(np.asarray([grid[0][i, j], grid[1][i, j]]), axis=1).transpose()).cuda())
            val = surrogate(
                torch.tensor(np.expand_dims(np.asarray([grid[0][i, j], grid[1][i, j]]), axis=1).transpose()).to(
                    device=device))
            value_grid[i, j] = val.mean
            uncertainty_grid[i, j] = val.stddev

    return value_grid, uncertainty_grid


for i, surrogate in enumerate(likelihood.surrogates):
    value_grid, uncertainty_grid = grid_to_surrogate_2D(grid, surrogate)
    expt_value = dataplex.properties.properties[i]._value.m
    expt_uncertainty = dataplex.properties.properties[i]._uncertainty.m
    expt_pressure = dataplex.properties.properties[i].thermodynamic_state.pressure.m
    expt_temperature = dataplex.properties.properties[i].thermodynamic_state.temperature.m
    plt.contourf(grid[0], grid[1], abs(expt_value - value_grid), 20, cmap='RdGy')
    plt.colorbar()
    plt.scatter(params[:, 0], params[:, 1], marker='x', color='0', label='Differential Evolution')
    plt.scatter(params_gd[0], params_gd[1], marker='v', color='0', label='L-BFGS-B')
    plt.scatter(result_brute[0][0], result_brute[0][1], marker='s', color='0', label='Brute Force')
    plt.scatter(params_gd_ub[0], params_gd_ub[1], marker='P', color='0', label='BFGS')
    plt.xlabel('[#18:1] epsilon (kcal/mol)')
    plt.ylabel('[#18:1] rmin_half (angstroms)')
    plt.title(
        f'Argon density deviation from experiment (g/ml) \n (Experimental value = {expt_value} g/ml @ {expt_temperature} K, {expt_pressure} atm)')
    plt.savefig(os.path.join('result/figures', f'surrogate_values_{expt_temperature}_K_{expt_pressure}_atm.png'),
                dpi=300)
    plt.legend()
    plt.show()

    plt.contourf(grid[0], grid[1], uncertainty_grid, 20, cmap='RdGy')
    plt.colorbar()
    plt.scatter(dataplex.parameter_values.to_numpy()[:, 0], dataplex.parameter_values.to_numpy()[:, 1], color='1',
                marker='x')

    plt.xlabel('[#18:1] epsilon (kcal/mol)')
    plt.ylabel('[#18:1] rmin_half (angstroms)')
    plt.title(
        f'Argon density uncertainties (g/ml) \n (Experimental value {expt_uncertainty} g/ml @ {expt_temperature} K, {expt_pressure} atm)')
    # plt.title('Latin Hypercube Sampling of argon LJ parameters')
    plt.savefig(os.path.join('result/figures', f'surrogate_uncertainties_{expt_temperature}_K_{expt_pressure}_atm.png'),
                dpi=300)
    plt.show()
