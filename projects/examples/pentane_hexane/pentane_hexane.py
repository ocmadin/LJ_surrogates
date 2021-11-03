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
import textwrap

gc.collect()
torch.cuda.empty_cache()
device = torch.device('cpu')
path = '../../../data/pentane-hexane'
smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]']
forcefield = 'openff-1-3-0.offxml'
dataset_json = 'test-set-collection-pent-hex-density.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
# test_params_one = torch.tensor(test_params[:, 0].reshape(test_params[:, 0].shape[0], 1).transpose()).detach()
likelihood = likelihood_function(dataplex, device)

mcmc, initial_parameters = likelihood.sample(samples=5000, step_size=0.0001, max_tree_depth=5, num_chains=1)
params = mcmc.get_samples()['parameters'].cpu().flatten(end_dim=1).numpy()
params = np.append(params, initial_parameters.numpy(), axis=0)

# likelihood.evaluate_surrogate_gpflow(likelihood.surrogates[0],test_params)


os.makedirs(os.path.join('result','figures'),exist_ok=True)
np.save(os.path.join('result','params.npy'), params)
ranges = dataplex.export_sampling_ranges()
df = pandas.DataFrame(params[:-1], columns=likelihood.flat_parameter_names)
wrapper = textwrap.TextWrapper(width=25)
columns = {}
for i, column in enumerate(df.columns):
    columns[column] = wrapper.fill(column)
df.rename(columns=columns, inplace=True)

pairplot = seaborn.pairplot(df, kind='kde', corner=True)
pairplot.map_upper(seaborn.kdeplot, levels=4, color=".2")
plt.tight_layout()
pairplot.savefig(os.path.join('result/figures', 'trace.png'), dpi=300)
plot_triangle(params,likelihood,ranges)