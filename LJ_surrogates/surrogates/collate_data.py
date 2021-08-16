import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.toolkit.typing.engines.smirnoff import ForceField
import os
import json
import numpy as np


def collate_physical_property_data(directory, smirks):
    data = []
    for i in range(int(len(os.listdir(directory)) / 2)):
        if os.path.isfile(os.path.join(directory, 'force_field_' + str(i) + '.offxml')) and os.path.isfile(
                os.path.join(directory, 'estimated_data_set_' + str(i) + '.json')):
            forcefield = ForceField(os.path.join(directory, 'force_field_' + str(i) + '.offxml'))
            results = PhysicalPropertyDataSet.from_json(
                os.path.join(directory, 'estimated_data_set_' + str(i) + '.json'))
            parameters, values = get_force_field_parameters(forcefield, smirks)
            if len(results) != 0:
                data.append([results, [parameters, values]])
    properties_all = get_training_data(data)
    return properties_all


def get_force_field_parameters(forcefield, smirks):
    lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
    param_dict = {}
    for i in range(len(lj_params.parameters)):
        if lj_params[i].smirks in smirks:
            param_dict[lj_params[i].smirks + '_epsilon'] = lj_params[i].epsilon.value_in_unit(lj_params[i].epsilon.unit)
            param_dict[lj_params[i].smirks + '_rmin_half'] = lj_params[i].rmin_half.value_in_unit(
                lj_params[i].rmin_half.unit)
    return list(param_dict.keys()), list(param_dict.values())


def get_training_data(data):
    properties_all = []
    for property in data[0][0].properties:
        train_data = {}
        X = []
        Y = []
        for datum in data:
            X.append(np.asarray(datum[1][1]))
            for prop in datum[0].properties:
                if prop.substance == property.substance and type(prop) == type(property):
                    Y.append([prop.value.magnitude, prop.uncertainty.magnitude])
                    id = prop.substance.identifier
        train_data[id] = [np.asarray(datum[1][0]), np.asarray(X), np.asarray(Y)]
        properties_all.append(train_data)
    return properties_all
