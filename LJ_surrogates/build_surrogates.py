import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.toolkit.typing.engines.smirnoff import ForceField
import os
import json
import numpy as np

def collate_physical_property_data(directory, smirks):

    data = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,folder)):
            forcefield = ForceField(os.path.join(directory,folder,'force-field.offxml'))
            results = PhysicalPropertyDataSet.from_json(os.path.join(directory,folder,'estimated_data_set.json'))
            parameters, values = get_force_field_parameters(os.path.join(directory,folder),forcefield,smirks)
            data.append([results,[parameters,values]])
    properties_all = get_training_data(data)
    return properties_all
def get_force_field_parameters(path,forcefield,smirks):
    lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
    param_dict = {}
    for lj in lj_params:
        if lj.smirks in smirks:
            param_dict[lj.smirks+'_epsilon'] = lj.epsilon.value_in_unit(lj.epsilon.unit)
            param_dict[lj.smirks+'_rmin_half'] = lj.rmin_half.value_in_unit(lj.rmin_half.unit)
    with open(os.path.join(path,'lj_params.txt'), 'w') as convert_file:
        convert_file.write(json.dumps(param_dict))
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
                    Y.append([prop.value.magnitude,prop.uncertainty.magnitude])
                    id = prop.substance.identifier
        train_data[id] = [np.asarray(datum[1][0]),np.asarray(X),np.asarray(Y)]
        properties_all.append(train_data)
    return properties_all
