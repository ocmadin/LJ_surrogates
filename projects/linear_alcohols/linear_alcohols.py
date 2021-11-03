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
import textwrap
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

test_params = vary_parameters_lhc(forcefield, 2, '.', smirks_types_to_change,
                                  [[0.95, 1.25], [0.95, 1.05], [0.95, 1.65], [0.95, 1.05], [0.5, 1.05], [0.9, 1.1],
                                   [0.9, 1.1], [0.95, 1.05], [0.95, 1.05], [0.95, 1.05]],
                                  parameter_sets_only=True, nonuniform_ranges=True).transpose()
test_params_one = torch.tensor(test_params[:, 0].reshape(test_params[:, 0].shape[0], 1).transpose()).to(
    device=device).detach()
# test_params_one = torch.tensor(test_params[:, 0].reshape(test_params[:, 0].shape[0], 1).transpose()).detach()
likelihood = likelihood_function(dataplex, device)

start = time.time()
predict, stddev = likelihood.evaluate_parameter_set(test_params_one)
end = time.time()
print(f'GPyTorch: {end - start} seconds')
start = time.time()
predict_botorch, stddev_botorch = likelihood.evaluate_parameter_set_botorch(test_params_one)
end = time.time()
print(f'Botorch: {end - start} seconds')
start = time.time()
predict_botorch_multi, stddev_botorch_multi = likelihood.evaluate_parameter_set_multisurrogate(test_params_one)
end = time.time()
print(f'Botorch Multi: {end - start} seconds')
mcmc, initial_parameters = likelihood.sample(samples=1000, step_size=0.0001, max_tree_depth=5, num_chains=1)
# params = mcmc.get_samples()['parameters'].cpu().flatten(end_dim=1).numpy()
# params = np.append(params,initial_parameters.cpu().numpy(),axis=0)
params = mcmc.get_samples()['parameters'].cpu().flatten(end_dim=1).numpy()
params = np.append(params, initial_parameters.numpy(), axis=0)

# likelihood.evaluate_surrogate_gpflow(likelihood.surrogates[0],test_params)


os.makedirs(os.path.join('result', 'figures'), exist_ok=True)
np.save(os.path.join('result', 'params.npy'), params)
ranges = dataplex.export_sampling_ranges()
df = pandas.DataFrame(params[:-1], columns=likelihood.flat_parameter_names)
wrapper = textwrap.TextWrapper(width=25)
columns = {}
for i, column in enumerate(df.columns):
    columns[column] = wrapper.fill(column)
df.rename(columns=columns, inplace=True)

pairplot = seaborn.pairplot(df, kind='hist', corner=True)
pairplot.map_upper(seaborn.kdeplot, levels=4, color=".2")
plt.tight_layout()
pairplot.savefig(os.path.join('result/figures', 'trace_hist.png'), dpi=300)

if df.shape[0] < 10000:
    n = df.shape[0]
else:
    n = 10000

pairplot = seaborn.pairplot(df.sample(n=n), kind='kde', corner=True)
pairplot.map_upper(seaborn.kdeplot, levels=4, color=".2")
plt.tight_layout()
pairplot.savefig(os.path.join('result/figures', 'trace_subsample.png'), dpi=300)

plot_triangle(params, likelihood, ranges, n, None)
