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
path = '../../../data/argon-single-5-small'
smirks_types_to_change = ['[#18:1]']
forcefield = 'openff-1-3-0-argon.offxml'
dataset_json = 'argon_single.json'
device = 'cpu'

dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device)
dataplex.plot_properties()

from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(dataplex.multisurrogate, beta=0.1)

from botorch.optim import optimize_acqf

bounds = torch.stack([torch.tensor([0,0]), torch.tensor([0.5,3.0])])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
