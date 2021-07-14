from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from simtk import openmm, unit
from scipy.stats import distributions
import copy
import numpy as np
import os

from smt.sampling_methods import LHS

def vary_parameters_lhc(filename, num_samples, output_directory):
    forcefield = ForceField(filename, allow_cosmetic_attributes=True)
    lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)

    smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]', '[#8X2H1+0:1]', '[#1:1]-[#8]']

    param_range = np.asarray([0.75, 1.25])
    n_dim = len(smirks_types_to_change) * 2
    lj_sample_ranges = []
    for i in range(n_dim):
        lj_sample_ranges.append(param_range)
    lj_sample_ranges = np.asarray(lj_sample_ranges)
    sampling = LHS(xlimits=lj_sample_ranges)
    values = sampling(num_samples)
    os.makedirs(output_directory,exist_ok=True)
    for i, value in enumerate(values):
        reshape_values = value.reshape((int(n_dim/2), 2))
        counter = 0
        for lj in lj_params:
            if lj.smirks in smirks_types_to_change:
                lj.epsilon *= reshape_values[counter, 0]
                lj.rmin_half *= reshape_values[counter, 1]
                counter += 1
        os.makedirs(os.path.join(output_directory,str(i)))
        ff_name = 'openff-1-3-0-LHS-modified.offxml'
        forcefield.to_file(os.path.join(output_directory, str(i),ff_name))

