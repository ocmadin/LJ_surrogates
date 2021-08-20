from LJ_surrogates.surrogates.collate_data import collate_physical_property_data
import matplotlib.pyplot as plt
import numpy as np
import gpflow
from fffit.fffit.models import run_gpflow_scipy
from LJ_surrogates.parameter_modification import vary_parameters_lhc
import time
from LJ_surrogates.sampling.likelihood import likelihood_function


path = '/media/owenmadin/storage/alcohol_alkane/alcohol_alkane/8_18_40_runs'

smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]', '[#8X2H1+0:1]', '[#1:1]-[#8]']

dataplex = collate_physical_property_data(path, smirks_types_to_change,'test-set-collection.json')
test_params = vary_parameters_lhc('openff-1-3-0.offxml',100,'.',parameter_sets_only=True).transpose()
test_params_one = test_params[:,0].reshape(test_params[:,0].shape[0],1).transpose()
predictions = dataplex.evaluate_parameter_set(test_params_one)
likelihood = likelihood_function(dataplex)
likelihood.sample(samples=1000)


