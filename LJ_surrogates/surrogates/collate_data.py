import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.toolkit.typing.engines.smirnoff import ForceField
import os
import json
import numpy as np
import torch
from LJ_surrogates.surrogates.surrogate import GPSurrogateModel


def collate_physical_property_data(directory, smirks, initial_forcefield, properties_filepath):
    data = []
    for i in range(int(len(os.listdir(directory)) / 2)):
        if os.path.isfile(os.path.join(directory, 'force_field_' + str(i) + '.offxml')) and os.path.isfile(
                os.path.join(directory, 'estimated_data_set_' + str(i) + '.json')):
            forcefield = ForceField(os.path.join(directory, 'force_field_' + str(i) + '.offxml'))
            results = PhysicalPropertyDataSet.from_json(
                os.path.join(directory, 'estimated_data_set_' + str(i) + '.json'))
            parameters = get_force_field_parameters(forcefield, smirks)
            if len(results) != 0:
                data.append([results, parameters])
    initial_forcefield = ForceField(initial_forcefield)
    initial_parameters = get_force_field_parameters(initial_forcefield, smirks)
    properties = PhysicalPropertyDataSet.from_json(properties_filepath)
    dataplex = get_training_data_new(data, properties, initial_parameters)
    # properties_all = get_training_data(data)
    return dataplex


def get_force_field_parameters(forcefield, smirks):
    lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
    param_dict = {}
    for i in range(len(lj_params.parameters)):
        if lj_params[i].smirks in smirks:
            param_dict[lj_params[i].smirks] = [lj_params[i].epsilon, lj_params[i].rmin_half]
    return param_dict


class LJParameter:
    def __init__(self, identity, values):
        self.smirks = identity
        self.epsilon = values[0]
        self.rmin_half = values[1]


class ParameterSetData:
    def __init__(self, datum):
        self.property_measurements = datum[0]
        self.parameters = []
        for key in datum[1].keys():
            self.parameters.append(LJParameter(key, datum[1][key]))


class ParameterSetDataMultiplex:
    def __init__(self, ParameterDataSetList, InitialProperties, InitialParameters):
        self.multi_data = ParameterDataSetList
        self.parameters = self.multi_data[0].parameters
        self.properties = InitialProperties
        self.initial_parameters = InitialParameters

    def align_data(self):
        pass

    def check_parameters(self):
        multi_data = []
        equality = True
        for dataset in self.multi_data:
            for i in range(len(self.parameters)):
                if dataset.parameters[i].smirks != self.parameters[i].smirks:
                    equality = False
                    print('!!!')
            if equality is True:
                multi_data.append(dataset)
        self.multi_data = multi_data

    def check_properties(self):
        multi_data = []
        for dataset in self.multi_data:
            equality = True
            if len(self.properties) != len(dataset.property_measurements):
                equality = False
            else:
                for i in range(len(self.properties)):
                    if type(self.properties.properties[i]) != type(dataset.property_measurements.properties[i]) or \
                            self.properties.properties[i].substance != dataset.property_measurements.properties[
                        i].substance \
                            or self.properties.properties[i].thermodynamic_state != \
                            dataset.property_measurements.properties[i].thermodynamic_state:
                        equality = False
            if equality == True:
                multi_data.append(dataset)
        self.multi_data = multi_data

    def align_property_data(self):
        parameter_labels = []
        for parameter in self.parameters:
            parameter_labels.append(parameter.smirks + '_epsilon')
            parameter_labels.append(parameter.smirks + '_rmin_half')
        all_parameters = []
        for data in self.multi_data:
            parameters = []
            for parameter in data.parameters:
                parameters.append(parameter.epsilon._value)
                parameters.append(parameter.rmin_half._value)
            all_parameters.append(parameters)
        property_labels = []
        for property in self.properties.properties:
            property_type = str(type(property)).split(sep='.')[-1].rstrip("'>")
            property_labels.append(str(property.substance) + "_" + property_type)
        property_measurements = []
        property_uncertainties = []
        for data in self.multi_data:
            measurements = []
            uncertainties = []
            for property in data.property_measurements.properties:
                measurements.append(property.value.m)
                uncertainties.append(property.uncertainty.m)
            property_measurements.append(measurements)
            property_uncertainties.append(uncertainties)
        all_parameters = np.asarray(all_parameters)
        property_measurements = np.asarray(property_measurements)
        property_uncertainties = np.asarray(property_uncertainties)

        self.parameter_labels = parameter_labels
        self.property_labels = property_labels
        self.parameter_values = pandas.DataFrame(all_parameters, columns=parameter_labels)
        self.property_measurements = pandas.DataFrame(property_measurements, columns=property_labels)
        self.property_uncertainties = pandas.DataFrame(property_uncertainties, columns=property_labels)

    def build_surrogates(self):
        surrogates = []

        if self.property_measurements.shape[0] != self.parameter_values.shape[0]:
            raise ValueError('Number of Parameter Sets and Measurement Sets must agree!')
        else:
            num_surrogates = self.property_measurements.shape[1]
        surrogate_measurements = self.property_measurements.values.transpose()
        for i in range(num_surrogates):
            individual_property_measurements = surrogate_measurements[i].reshape(
                (surrogate_measurements[0].shape[0], 1))
            model = GPSurrogateModel(parameter_data=self.parameter_values.values,
                                     property_data=individual_property_measurements)
            model.build_surrogate_GPytorch()
            model.model.train_targets = model.model.train_targets.detach()
            surrogates.append(model)
        self.surrogates = surrogates

    def evaluate_parameter_set(self, parameter_set):

        predictions_all = []
        uncertainties_all = []
        for surrogate in self.surrogates:
            predictions = surrogate.likelihood(surrogate.model(torch.tensor(parameter_set)))
            predictions_all.append(predictions.mean)
            uncertainties_all.append(predictions.stddev)
        uncertainties_all = torch.cat(uncertainties_all)
        predictions_all = torch.cat(predictions_all)
        uncertainties_all = uncertainties_all.reshape(uncertainties_all.shape[0],1)
        predictions_all = predictions_all.reshape(predictions_all.shape[0],1)
        return predictions_all, uncertainties_all
        return predictions_all
def get_training_data_new(data, properties, parameters):
    data_list = []
    for datum in data:
        data_list.append(ParameterSetData(datum))
    dataplex = ParameterSetDataMultiplex(data_list, properties, parameters)
    dataplex.check_parameters()
    dataplex.check_properties()
    dataplex.align_property_data()
    dataplex.build_surrogates()
    return dataplex


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
