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
gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/alcohol_alkane/argon_all'



# smirks_types_to_change = ['[#6X4:1]']
smirks_types_to_change = ['[#18:1]']
# smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]', '[#8X2H1+0:1]', '[#1:1]-[#8]']

dataplex = collate_physical_property_data(path, smirks_types_to_change, 'openff-1-3-0-argon.offxml', 'argon_all.json')
test_params = vary_parameters_lhc('openff-1-3-0.offxml',2,'.',smirks_types_to_change, [0.9,1.1],  parameter_sets_only=True).transpose()
test_params_one = torch.tensor(test_params[:,0].reshape(test_params[:,0].shape[0],1).transpose()).to(device=device).detach()
grid = create_evaluation_grid('openff-1-3-0-argon.offxml',smirks_types_to_change,np.array([0.75,1.25]))


#predictions = dataplex.evaluate_parameter_set(test_params_one)

likelihood = likelihood_function(dataplex)


def grid_to_surrogate_2D(grid,surrogate):
    value_grid = np.empty((grid[0].shape[0],grid[0].shape[1]))
    uncertainty_grid = np.empty((grid[0].shape[0],grid[0].shape[1]))

    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            val = surrogate.likelihood(surrogate.model(torch.tensor(np.expand_dims(np.asarray([grid[0][i,j],grid[1][i,j]]),axis=1).transpose()).cuda()))
            value_grid[i,j] = val.mean
            uncertainty_grid[i,j] = val.stddev

    return value_grid, uncertainty_grid



value_grid,uncertainty_grid = grid_to_surrogate_2D(grid,likelihood.surrogates[0])
start = time.time()
predict, stddev = likelihood.evaluate_parameter_set(test_params_one)
end = time.time()
duration = end - start
start = time.time()
predictions = likelihood.evaluate_parameter_set(test_params_one)
end = time.time()
print(f'With map: {end-start} seconds')
start = time.time()
predictions_map = likelihood.evaluate_parameter_set_map(test_params_one)
end = time.time()
print(f'Without map: {end-start} seconds')
mcmc = likelihood.sample(samples=5000)
# with open('mcmc_result.pickle', 'wb') as f:
#     pickle.dump(mcmc, f)
params = mcmc.get_samples()['parameters'].cpu().flatten(end_dim=1).numpy()

# likelihood.evaluate_surrogate_gpflow(likelihood.surrogates[0],test_params)

df = pandas.DataFrame(params, columns=likelihood.flat_parameter_names)
pairplot = seaborn.pairplot(df, kind='kde', corner=True)
pairplot.map_upper(seaborn.kdeplot,levels=4, color=".2")
plt.show()
pairplot.savefig('trace.png',dpi=300)


plt.contourf(grid[0],grid[1],value_grid,20, cmap='RdGy')
plt.colorbar()




plt.xlabel('[#6X4:1] epsilon (kcal/mol)')
plt.ylabel('[#6X4:1] rmin_half (angstroms)')
plt.title('Argon density (g/ml) \n (Experimental value = 1.41 g/ml @ 89.13 K, 97.44 atm)')
plt.savefig('surrogate_values.png',dpi=300)
plt.show()


plt.contourf(grid[0],grid[1],uncertainty_grid,20, cmap='RdGy')
plt.colorbar()
plt.scatter(dataplex.parameter_values.to_numpy()[:,0], dataplex.parameter_values.to_numpy()[:,1], color='1',
            marker='x')


plt.xlabel('[#18:1] epsilon (kcal/mol)')
plt.ylabel('[#18:1] rmin_half (angstroms)')
plt.title('Argon density uncertainties (g/ml) \n (Experimental value 0.007 g/ml @ 89.13 K, 97.44 atm')
plt.savefig('surrogate_uncertainties.png',dpi=300)
plt.show()
