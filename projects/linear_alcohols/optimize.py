from LJ_surrogates.sampling.optimize import ForceBalanceObjectiveFunction, create_forcefields_from_optimized_params
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
import torch
import gc
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import numpy as np
import pandas
import textwrap
import seaborn

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

objective = ForceBalanceObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                                 dataplex.property_labels)
objective.flatten_parameters()
bounds = []
for column in dataplex.parameter_values.columns:
    minbound = min(dataplex.parameter_values[column].values)
    maxbound = max(dataplex.parameter_values[column].values)
    bounds.append((minbound, maxbound))

objs = []
params = []
objs_l_bfgs_b = []
params_l_bfgs_b = []
for i in range(5):
    result_l_bfgs_b = minimize(objective,objective.flat_parameters, bounds=bounds, method='L-BFGS-B')
    result_de = differential_evolution(objective, bounds)
    objs.append(result_de.fun)
    params.append(result_de.x)
    objs_l_bfgs_b.append(result_l_bfgs_b.fun)
    params_l_bfgs_b.append(result_l_bfgs_b.x)

samples = np.load('result_500k_10_25/params.npy')[::100]

df = pandas.DataFrame(params[:-1], columns=objective.flat_parameter_names)
df2 = pandas.DataFrame(np.expand_dims(params[-1],axis=0), columns=objective.flat_parameter_names)
wrapper = textwrap.TextWrapper(width=25)
columns = {}
for i, column in enumerate(df.columns):
    columns[column] = wrapper.fill(column)
df.rename(columns=columns, inplace=True)
pairplot = seaborn.pairplot(df, kind='kde', corner=True)
for i in range(pairplot.axes.shape[0]):
    for j in range(pairplot.axes.shape[0]):
        if i == j:
            for param_set in params:
                pairplot.axes[i][j].axvline(param_set[i], color='k')
            for param_set in params_l_bfgs_b:
                pairplot.axes[i][j].axvline(param_set[i], ls='--', color='k')
        elif i > j:
            for param_set in params:
                pairplot.axes[i][j].scatter(param_set[j], param_set[i], marker='x', color='k')
            for param_set in params_l_bfgs_b:
                pairplot.axes[i][j].scatter(param_set[j], param_set[i], marker='+', color='k')
plt.tight_layout()
pairplot.savefig('trace_with_opt.png', dpi=300)


create_forcefields_from_optimized_params(params,objective.flat_parameter_names,'openff-1-3-0.offxml')
