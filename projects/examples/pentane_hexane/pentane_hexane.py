import torch.cuda

from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
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
from LJ_surrogates.plotting.plotting import plot_triangle

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '../../../data/pentane-hexane'
smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = 'test-set-collection-pent-hex-density.json'


dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json)
test_params = vary_parameters_lhc(forcefield, 2, '.', smirks_types_to_change, [0.9, 1.1],
                                  parameter_sets_only=True).transpose()
test_params_one = torch.tensor(test_params[:, 0].reshape(test_params[:, 0].shape[0], 1).transpose()).to(
    device=device).detach()
grid = create_evaluation_grid(forcefield, smirks_types_to_change, np.array([0.75, 1.25]))
likelihood = likelihood_function(dataplex)

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
mcmc = likelihood.sample(samples=1000)
params = mcmc.get_samples()['parameters'].cpu().flatten(end_dim=1).numpy()


# likelihood.evaluate_surrogate_gpflow(likelihood.surrogates[0],test_params)


os.makedirs(os.path.join('result','figures'),exist_ok=True)
np.save(os.path.join('result','params.npy'), params)

ranges = dataplex.export_sampling_ranges()
plot_triangle(params,likelihood,ranges)