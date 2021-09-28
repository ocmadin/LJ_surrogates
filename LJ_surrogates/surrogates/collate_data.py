import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.toolkit.typing.engines.smirnoff import ForceField
import os
import json
import numpy as np
import torch
from LJ_surrogates.surrogates.surrogate import GPSurrogateModel, build_surrogate_lightweight, build_surrogates_loo_cv
import matplotlib.pyplot as plt
import tqdm
import copy


def collate_physical_property_data(directory, smirks, initial_forcefield, properties_filepath):
    data = []
    counter = 0
    for i in range(int(len(os.listdir(directory)) / 2)):
        if os.path.isfile(os.path.join(directory, 'force_field_' + str(i) + '.offxml')) and os.path.isfile(
                os.path.join(directory, 'estimated_data_set_' + str(i) + '.json')):
            forcefield = ForceField(os.path.join(directory, 'force_field_' + str(i) + '.offxml'))
            results = PhysicalPropertyDataSet.from_json(
                os.path.join(directory, 'estimated_data_set_' + str(i) + '.json'))
            parameters = get_force_field_parameters(forcefield, smirks)
            if len(results) != 0:
                data.append([results, parameters])
    print(f'Started with {i+1} datasets, removed {i+1-len(data)} empty dataset(s)')
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
            if equality is True:
                multi_data.append(dataset)
        self.multi_data = multi_data

    def check_properties(self):
        multi_data = []
        self.properties = canonicalize_dataset(self.properties)
        for dataset in self.multi_data:
            dataset.property_measurements = canonicalize_dataset(dataset.property_measurements)
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

    def prune_bad_densities(self):
        print('Eliminating Bad Density Measurements...')
        before = len(self.multi_data)
        for i,property in enumerate(self.properties.properties):
            if self.property_labels[i].endswith('Density'):
                to_pop = []
                for j,measurement in enumerate(self.multi_data):
                    if measurement.property_measurements.properties[i].value.m <= 0.1 * property.value.m:
                        to_pop.append(j)
                for pop in sorted(to_pop,reverse=True):
                    del self.multi_data[pop]
        after = len(self.multi_data)
        print(f"Removed {before-after} datasets due to bad density measurments")
    def align_property_data(self):
        parameter_labels = []
        for parameter in self.parameters:
            parameter_labels.append(parameter.smirks + '_epsilon')
            parameter_labels.append(parameter.smirks + '_rmin_half')
        property_labels = []
        for property in self.properties.properties:
            property_type = str(type(property)).split(sep='.')[-1].rstrip("'>")
            property_labels.append(str(property.substance) + "_" + property_type)
        self.parameter_labels = parameter_labels
        self.property_labels = property_labels
        self.prune_bad_densities()
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
        all_parameters = []
        for data in self.multi_data:
            parameters = []
            for parameter in data.parameters:
                parameters.append(parameter.epsilon._value)
                parameters.append(parameter.rmin_half._value)
            all_parameters.append(parameters)
        all_parameters = np.asarray(all_parameters)
        property_measurements = np.asarray(property_measurements)
        property_uncertainties = np.asarray(property_uncertainties)


        self.parameter_values = pandas.DataFrame(all_parameters, columns=parameter_labels)
        self.property_measurements = pandas.DataFrame(property_measurements, columns=property_labels)
        self.property_uncertainties = pandas.DataFrame(property_uncertainties, columns=property_labels)

    def build_surrogates(self,do_cross_validation=True):
        surrogates = []

        if self.property_measurements.shape[0] != self.parameter_values.shape[0]:
            raise ValueError('Number of Parameter Sets and Measurement Sets must agree!')
        else:
            num_surrogates = self.property_measurements.shape[1]
        surrogate_measurements = self.property_measurements.values.transpose()
        surrogate_uncertainties = self.property_uncertainties.values.transpose()
        for i in tqdm.tqdm(range(num_surrogates),ascii=True,desc='Building and Validating Surrogates'):
            individual_property_measurements = surrogate_measurements[i].reshape(
                (surrogate_measurements[0].shape[0], 1))
            individual_property_uncertainties = surrogate_uncertainties[i].reshape(
                (surrogate_uncertainties[0].shape[0], 1))
            # model = GPSurrogateModel(parameter_data=self.parameter_values.values,
            #                          property_data=individual_property_measurements)
            # model.build_surrogate_gpflow()
            # model.model.train_targets = model.model.train_targets.detach()
            model = build_surrogate_lightweight(self.parameter_values.values,individual_property_measurements, individual_property_uncertainties)
            if do_cross_validation is True:
                build_surrogates_loo_cv(self.parameter_values.values,individual_property_measurements,individual_property_uncertainties,self.property_labels[i])
            model.train_targets = model.train_targets.detach()
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
        uncertainties_all = uncertainties_all.reshape(uncertainties_all.shape[0], 1)
        predictions_all = predictions_all.reshape(predictions_all.shape[0], 1)
        return predictions_all, uncertainties_all
        return predictions_all

    def export_sampling_ranges(self):
        params = self.parameter_values.to_numpy()
        ranges = []
        for i in range(params.shape[1]):
            ranges.append([min(params[:,i]),max(params[:,i])])

        ranges = np.asarray(ranges)
        return ranges

    def plot_properties(self):
        os.makedirs(os.path.join('result','properties','figures'), exist_ok=True)
        x_range = np.linspace(0,self.property_measurements.shape[0]-1,num=self.property_measurements.shape[0])
        for i, column in enumerate(self.property_measurements.columns):
            plt.errorbar(x_range, self.property_measurements[column].values, yerr=self.property_uncertainties[column].values, ls='none', capsize=0, marker='x')
            plt.axhline(self.properties.properties[i].value.m)
            plt.title(
                f'{str(self.properties.properties[i].substance)}: {self.properties.properties[i].value} +/- {self.properties.properties[i].uncertainty}')
            plt.xlabel('Parameter Set')
            plt.ylabel('Property Value')
            label = f'{str(self.properties.properties[i].substance)}_{self.properties.properties[i].value}'
            plt.savefig(os.path.join('result','properties','figures','property_'+str(i)+'.png'), dpi=300)
            plt.clf()


def get_training_data_new(data, properties, parameters):

    data_list = []
    for datum in data:
        data_list.append(ParameterSetData(datum))
    print('Collecting and Preparing Data...')
    dataplex = ParameterSetDataMultiplex(data_list, properties, parameters)
    before = copy.deepcopy(len(dataplex.multi_data))
    dataplex.check_parameters()
    dataplex.check_properties()
    after = len(dataplex.multi_data)
    print(f'Removed {before-after} incomplete or errored datasets')
    dataplex.align_property_data()
    dataplex.plot_properties()
    print(f'Proceeding to build surrogates with {len(dataplex.multi_data)} Datasets')
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


def canonicalize_dataset(dataset):
    if not isinstance(dataset, PhysicalPropertyDataSet):
        raise TypeError('Dataset must be a PhysicalPropertyDataSet object')
    ids = []
    for property in dataset.properties:
        ids.append(int(property.id, 16))
    ids = sorted(ids)
    for i, id in enumerate(ids):
        ids[i] = format(id, 'x')
    properties = []
    for id in ids:
        for property in dataset.properties:
            if property.id == id:
                properties.append(property)
    dataset._properties = tuple(properties)
    return dataset
