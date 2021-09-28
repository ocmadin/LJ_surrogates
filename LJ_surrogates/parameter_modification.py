from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from simtk import openmm, unit
from scipy.stats import distributions
import copy
import numpy as np
import os
import copy

from smt.sampling_methods import LHS


def vary_parameters_lhc(filename, num_samples, output_directory, smirks_types_to_change, param_range,
                        parameter_sets_only=False, nonuniform_ranges=False):
    forcefield = ForceField(filename, allow_cosmetic_attributes=True)

    # smirks_types_to_change = ['[#6X4:1]']
    # smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]']
    # smirks_types_to_change = ['[#6X4:1]', '[#1:1]-[#6X4]', '[#8X2H1+0:1]', '[#1:1]-[#8]']
    n_dim = len(smirks_types_to_change) * 2
    if nonuniform_ranges is True:
        lj_sample_ranges = np.asarray(param_range)
    else:
        lj_sample_ranges = []
        for i in range(n_dim):
            lj_sample_ranges.append(param_range)
        lj_sample_ranges = np.asarray(lj_sample_ranges)
    sampling = LHS(xlimits=lj_sample_ranges)
    values = sampling(num_samples)
    os.makedirs(output_directory, exist_ok=True)
    all_params = []
    for i, value in enumerate(values):
        ff_copy = copy.deepcopy(forcefield)
        lj_params = ff_copy.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
        params = []
        reshape_values = value.reshape((int(n_dim / 2), 2))
        counter = 0
        for lj in lj_params:
            if lj.smirks in smirks_types_to_change:
                if nonuniform_ranges is True:
                    lj.epsilon = reshape_values[counter, 0] * unit.kilocalorie_per_mole
                    lj.rmin_half = reshape_values[counter, 1] * unit.angstrom
                else:
                    lj.epsilon *= reshape_values[counter, 0]
                    lj.rmin_half *= reshape_values[counter, 1]
                params.append(lj.epsilon._value)
                params.append(lj.rmin_half._value)
                counter += 1
        all_params.append(params)
        if parameter_sets_only is False:
            os.makedirs(os.path.join(output_directory, str(i + 1)))
            ff_name = 'force-field.offxml'
            ff_copy.to_file(os.path.join(output_directory, str(i + 1), ff_name))
    if parameter_sets_only is True:
        return np.asarray(all_params)


def create_evaluation_grid(filename, smirks_types_to_change, param_range):
    forcefield = ForceField(filename, allow_cosmetic_attributes=True)
    lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
    n_dim = len(smirks_types_to_change) * 2
    lj_sample_ranges = []
    for i in range(n_dim):
        lj_sample_ranges.append(param_range)
    lj_sample_ranges = np.asarray(lj_sample_ranges)
    ranges = []
    for lj in lj_params:
        if lj.smirks in smirks_types_to_change:
            vals_eps = lj.epsilon._value * param_range
            vals_rmin = lj.rmin_half._value * param_range
            lin_eps = np.linspace(vals_eps[0], vals_eps[1], num=100)
            lin_rmin = np.linspace(vals_rmin[0], vals_rmin[1], num=100)
            ranges.append(lin_eps)
            ranges.append(lin_rmin)
    grid = np.meshgrid(*ranges)
    return grid
