from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
from LJ_surrogates.parameter_modification import vary_parameters_lhc
import time
from LJ_surrogates.sampling.likelihood import likelihood_function
import torch


device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/alcohol_alkane/8_18_40_runs'

smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]', '[#8X2H1+0:1]', '[#1:1]-[#8]']

dataplex = collate_physical_property_data(path, smirks_types_to_change,'test-set-collection.json')
test_params = vary_parameters_lhc('openff-1-3-0.offxml',100,'.',parameter_sets_only=True).transpose()
test_params_one = torch.tensor(test_params[:,0].reshape(test_params[:,0].shape[0],1).transpose()).to(device=device).detach()

#predictions = dataplex.evaluate_parameter_set(test_params_one)

likelihood = likelihood_function(dataplex)
start = time.time()
predict, stddev = likelihood.evaluate_parameter_set(test_params_one)
end = time.time()
duration = end - start
start = time.time()
map_predict, map_stddev = likelihood.evaluate_parameter_set_map(test_params_one)
end = time.time()
duration_map = end - start
print(duration)
print(duration_map)
likelihood.sample(samples=1000)


