import torch.cuda

from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
from LJ_surrogates.parameter_modification import vary_parameters_lhc, create_evaluation_grid
import time
from LJ_surrogates.sampling.likelihood import likelihood_function
from LJ_surrogates.surrogates.surrogate import compute_surrogate_gradients
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
from pyro.infer.mcmc.util import summary

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '../../../data/argon-single-50-small'
smirks_types_to_change = ['[#18:1]']
forcefield = 'openff-1-3-0-argon.offxml'
dataset_json = 'argon_single.json'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json)
dataplex.plot_properties()
test_params = vary_parameters_lhc(forcefield, 20, '.', smirks_types_to_change, [[0.18062470551820903, 0.48984215139407744], [1.836245579044131, 2.039671726465781]],
                                  parameter_sets_only=True,nonuniform_ranges=True).transpose()
test_params_one = torch.tensor(test_params[:, 0].reshape(test_params[:, 0].shape[0], 1).transpose()).to(
    device=device).detach()
grid = create_evaluation_grid(forcefield, smirks_types_to_change, np.array([0.75, 1.25]))
likelihood = likelihood_function(dataplex)


def grid_to_surrogate_2D(grid, surrogate):
    value_grid = np.empty((grid[0].shape[0], grid[0].shape[1]))
    uncertainty_grid = np.empty((grid[0].shape[0], grid[0].shape[1]))

    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            val = surrogate(
                torch.tensor(np.expand_dims(np.asarray([grid[0][i, j], grid[1][i, j]]), axis=1).transpose()).cuda())
            value_grid[i, j] = val.mean
            uncertainty_grid[i, j] = val.stddev

    return value_grid, uncertainty_grid


start = time.time()
predict, stddev = likelihood.evaluate_parameter_set(test_params_one)
end = time.time()
duration = end - start
start = time.time()
predictions = likelihood.evaluate_parameter_set(test_params_one)
end = time.time()
print(f'With map: {end - start} seconds')
start = time.time()
predictions_map = likelihood.evaluate_parameter_set_map(test_params_one)
end = time.time()
print(f'Without map: {end - start} seconds')
mcmc, initial_parameters = likelihood.sample(samples=1000, step_size=0.001, max_tree_depth=5, num_chains=1)
mcmc._samples['parameters'] = mcmc._samples['parameters'].cpu()
summary = summary(mcmc._samples)
params = mcmc.get_samples()['parameters'].cpu().flatten(end_dim=1).numpy()
params = np.append(params,initial_parameters.cpu().numpy(),axis=0)
ranges = dataplex.export_sampling_ranges()
sampled_ranges = []
for i in range(params.shape[1]):
    low_index = int(np.floor(0.025*len(params[:,i])))
    high_index = int(np.floor(0.975*len(params[:,i])))
    sorted_samples = sorted(params[:,i])
    sampled_ranges.append([sorted_samples[low_index],sorted_samples[high_index]])


# likelihood.evaluate_surrogate_gpflow(likelihood.surrogates[0],test_params)
os.makedirs(os.path.join('result', 'figures'), exist_ok=True)
np.save(os.path.join('result', 'params.npy'), params)
plot_triangle(params, likelihood, ranges)

gradients = compute_surrogate_gradients(dataplex.surrogates[0],dataplex.parameter_values.to_numpy()[0],0.01)
grad_vs_range = []
dist_vs_samples = []
for i in range(len(gradients)):
    grad_vs_range.append(abs(gradients[i].detach().cpu().numpy())/(ranges[i][1] - ranges[i][0]))
    dist_vs_samples.append((sampled_ranges[i][1] - sampled_ranges[i][0])/(ranges[i][1] - ranges[i][0]))

plt.clf()
for i, surrogate in enumerate(likelihood.surrogates):
    value_grid, uncertainty_grid = grid_to_surrogate_2D(grid, surrogate)
    expt_value = dataplex.properties.properties[i]._value.m
    expt_uncertainty = dataplex.properties.properties[i]._uncertainty.m
    expt_pressure = dataplex.properties.properties[i].thermodynamic_state.pressure.m
    expt_temperature = dataplex.properties.properties[i].thermodynamic_state.temperature.m
    plt.contourf(grid[0], grid[1], abs(expt_value - value_grid), 20, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('[#18:1] epsilon (kcal/mol)')
    plt.ylabel('[#18:1] rmin_half (angstroms)')
    plt.title(
        f'Argon density deviation from experiment (g/ml) \n (Experimental value = {expt_value} g/ml @ {expt_temperature} K, {expt_pressure} atm)')
    plt.savefig(os.path.join('result/figures', f'surrogate_values_{expt_temperature}_K_{expt_pressure}_atm.png'),
                dpi=300)
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
