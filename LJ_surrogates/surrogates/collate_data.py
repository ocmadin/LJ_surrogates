import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import enthalpy, density
from openff.toolkit.typing.engines.smirnoff import ForceField
import os
import numpy as np
import torch
from LJ_surrogates.surrogates.surrogate import build_surrogate_lightweight, build_surrogates_loo_cv, \
    build_surrogate_lightweight_botorch, build_multisurrogate_lightweight_botorch, \
    build_multisurrogate_independent_botorch, build_surrogates_loo_cv_independent
import matplotlib.pyplot as plt
import tqdm
import copy
import gpytorch


def collate_physical_property_data(directory, smirks, initial_forcefield, properties_filepath, device):
    data = []
    for i in range(int(len(os.listdir(directory)) / 2)):
        if os.path.isfile(os.path.join(directory, 'force_field_' + str(i + 1) + '.offxml')) and os.path.isfile(
                os.path.join(directory, 'estimated_data_set_' + str(i + 1) + '.json')):
            forcefield = ForceField(os.path.join(directory, 'force_field_' + str(i + 1) + '.offxml'))
            results = PhysicalPropertyDataSet.from_json(
                os.path.join(directory, 'estimated_data_set_' + str(i + 1) + '.json'))
            parameters = get_force_field_parameters(forcefield, smirks)
            if len(results.estimated_properties) != 0:
                data.append([results.estimated_properties, parameters])
            # if len(results) != 0:
            #     data.append([results, parameters])
    print(f'Started with {i + 1} datasets, removed {i + 1 - len(data)} empty dataset(s)')
    initial_forcefield = ForceField(initial_forcefield)
    initial_parameters = get_force_field_parameters(initial_forcefield, smirks)
    properties = PhysicalPropertyDataSet.from_json(properties_filepath)
    dataplex = get_training_data_new(data, properties, initial_parameters, device)
    # dataplex.plot_properties()
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
    def __init__(self, ParameterDataSetList, InitialProperties, InitialParameters, device):
        self.multi_data = ParameterDataSetList
        self.parameters = self.multi_data[0].parameters
        self.properties = InitialProperties
        self.initial_parameters = InitialParameters
        self.device = device

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
        failed_data = []
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
            else:
                failed_data.append(dataset)
        self.multi_data = multi_data
        self.failed_data = failed_data

    def prune_bad_densities(self):
        print('Eliminating Bad Density Measurements...')
        before = len(self.multi_data)
        self.bad_density_data = []
        for i, property in enumerate(self.properties.properties):
            if self.property_labels[i].endswith('Density'):
                to_pop = []
                for j, measurement in enumerate(self.multi_data):
                    if measurement.property_measurements.properties[i].value.m <= 0.1 * property.value.m:
                        to_pop.append(j)
                for pop in sorted(to_pop, reverse=True):
                    self.bad_density_data.append(self.multi_data[pop])
                    del self.multi_data[pop]
        after = len(self.multi_data)
        print(f"Removed {before - after} datasets due to bad density measurments")

    def prune_low_aa_hvaps(self):
        print('Removing Low Energy Acetic Acid meaasurements')
        before = len(self.multi_data)

        self.bad_hvap_data = []
        for i, property in enumerate(self.properties.properties):
            if self.property_labels[i] == 'CC(=O)O{solv}{x=1.000000}_EnthalpyOfVaporization':
                to_pop = []
                for j, measurement in enumerate(self.multi_data):
                    if measurement.property_measurements.properties[i].value.m <= 60:
                        to_pop.append(j)
                for pop in sorted(to_pop, reverse=True):
                    self.bad_hvap_data.append(self.multi_data[pop])
                    del self.multi_data[pop]
        after = len(self.multi_data)
        print(f"Removed {before - after} datasets due to low acetic acid hvap measurments")

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
        # self.prune_low_aa_hvaps()
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
        failed_parameters = []
        bad_density_parameters = []
        for data in self.multi_data:
            parameters = []
            for parameter in data.parameters:
                parameters.append(parameter.epsilon._value)
                parameters.append(parameter.rmin_half._value)
            all_parameters.append(parameters)
        for data in self.failed_data:
            parameters = []
            for parameter in data.parameters:
                parameters.append(parameter.epsilon._value)
                parameters.append(parameter.rmin_half._value)
            failed_parameters.append(parameters)
        for data in self.bad_density_data:
            parameters = []
            for parameter in data.parameters:
                parameters.append(parameter.epsilon._value)
                parameters.append(parameter.rmin_half._value)
            bad_density_parameters.append(parameters)
        all_parameters = np.asarray(all_parameters)
        failed_parameters = np.asarray(failed_parameters)
        bad_density_parameters = np.asarray(bad_density_parameters)
        property_measurements = np.asarray(property_measurements)
        property_uncertainties = np.asarray(property_uncertainties)
        if len(failed_parameters) > 0:
            self.failed_params_values = pandas.DataFrame(failed_parameters, columns=parameter_labels)
        self.parameter_values = pandas.DataFrame(all_parameters, columns=parameter_labels)
        if len(bad_density_parameters) > 0:
            self.bad_density_param_values = pandas.DataFrame(bad_density_parameters, columns=parameter_labels)
        # self.plot_parameter_sets()
        self.property_measurements = pandas.DataFrame(property_measurements, columns=property_labels)
        self.property_uncertainties = pandas.DataFrame(property_uncertainties, columns=property_labels)

    def build_surrogates(self, do_cross_validation=False):
        surrogates = []
        botorch_surrogates = []

        if self.property_measurements.shape[0] != self.parameter_values.shape[0]:
            raise ValueError('Number of Parameter Sets and Measurement Sets must agree!')
        else:
            num_surrogates = self.property_measurements.shape[1]
        surrogate_measurements = self.property_measurements.values.transpose()
        surrogate_uncertainties = self.property_uncertainties.values.transpose()
        for i in tqdm.tqdm(range(num_surrogates), ascii=True, desc='Building and Validating Surrogates'):
            individual_property_measurements = surrogate_measurements[i].reshape(
                (surrogate_measurements[0].shape[0], 1))
            individual_property_uncertainties = surrogate_uncertainties[i].reshape(
                (surrogate_uncertainties[0].shape[0], 1))
            model = build_surrogate_lightweight(self.parameter_values.values, individual_property_measurements,
                                                individual_property_uncertainties, self.device)
            botorch_model = build_surrogate_lightweight_botorch(self.parameter_values.values,
                                                                individual_property_measurements,
                                                                individual_property_uncertainties, self.device)
            if do_cross_validation is True:
                build_surrogates_loo_cv(self.parameter_values.values, surrogate_measurements,
                                        surrogate_uncertainties, self.property_labels, self.parameter_labels)
            model.train_targets = model.train_targets.detach()
            surrogates.append(model)
            botorch_surrogates.append(botorch_model)
        self.surrogates = surrogates
        self.botorch_surrogates = botorch_surrogates

    def build_multisurrogates(self, do_cross_validation=False):

        if self.property_measurements.shape[0] != self.parameter_values.shape[0]:
            raise ValueError('Number of Parameter Sets and Measurement Sets must agree!')
        else:
            num_surrogates = self.property_measurements.shape[1]
        surrogate_measurements = self.property_measurements.values.transpose()
        surrogate_uncertainties = self.property_uncertainties.values.transpose()
        self.multisurrogate = build_multisurrogate_independent_botorch(self.parameter_values.values,
                                                                       surrogate_measurements,
                                                                       surrogate_uncertainties, self.device)
        if do_cross_validation is True:
            build_surrogates_loo_cv_independent(self.parameter_values.values, surrogate_measurements,
                                                surrogate_uncertainties, self.property_labels, self.parameter_labels)

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
            ranges.append([min(params[:, i]), max(params[:, i])])

        ranges = np.asarray(ranges)
        return ranges

    def plot_parameter_sets(self, show=False):
        all_params = []
        df_1 = copy.deepcopy(self.parameter_values)
        import textwrap
        import seaborn
        wrapper = textwrap.TextWrapper(width=25)
        columns = {}
        for i, column in enumerate(df_1.columns):
            columns[column] = wrapper.fill(column)
        df_1.rename(columns=columns, inplace=True)
        df_1['status'] = np.tile(['completed'], df_1.shape[0])
        all_params.append(df_1)
        if hasattr(self, 'failed_params_values'):
            df_2 = self.failed_params_values
            wrapper = textwrap.TextWrapper(width=25)
            columns = {}
            for i, column in enumerate(df_2.columns):
                columns[column] = wrapper.fill(column)
            df_2.rename(columns=columns, inplace=True)
            df_2['status'] = np.tile(['failed'], df_2.shape[0])
            all_params.append(df_2)
        if hasattr(self, 'bad_density_param_values'):
            df_3 = self.bad_density_param_values
            wrapper = textwrap.TextWrapper(width=25)
            columns = {}
            for i, column in enumerate(df_3.columns):
                columns[column] = wrapper.fill(column)
            df_3.rename(columns=columns, inplace=True)
            df_3['status'] = np.tile(['low density'], df_3.shape[0])
            all_params.append(df_3)
        df = pandas.concat(all_params, axis=0, ignore_index=True)
        pairplot = seaborn.pairplot(df, corner=True, plot_kws=dict(marker="x", linewidth=3), hue='status',
                                    diag_kind="hist")
        os.makedirs(os.path.join('result', 'properties', 'figures'), exist_ok=True)
        pairplot.savefig(os.path.join('result', 'properties', 'figures', 'params.png'), dpi=300)
        if show is True:
            plt.show()
        plt.close()

    def plot_properties(self, show=False):
        os.makedirs(os.path.join('result', 'properties', 'figures'), exist_ok=True)
        x_range = np.linspace(0, self.property_measurements.shape[0] - 1, num=self.property_measurements.shape[0])

        for i, column in enumerate(self.property_measurements.columns):
            plt.errorbar(x_range, self.property_measurements[column].values,
                         yerr=self.property_uncertainties[column].values, ls='none', capsize=0, marker='x', color='k',
                         label="Simulated Measurements")
            plt.axhline(self.properties.properties[i].value.m, label='Experimental value')
            plt.axhspan(self.properties.properties[i].value.m - self.properties.properties[i].uncertainty.m,
                        self.properties.properties[i].value.m + self.properties.properties[i].uncertainty.m, 0, 1,
                        color='b', alpha=0.1, label='Experimental Uncertainty')
            plt.legend()
            plt.title(
                f'{str(self.properties.properties[i].substance)}: {self.properties.properties[i].value} +/- {self.properties.properties[i].uncertainty}')
            plt.xlabel('Parameter Set')
            plt.ylabel('Property Value')
            # label = f'{str(self.properties.properties[i].substance)}_{self.properties.properties[i].value}'
            plt.savefig(os.path.join('result', 'properties', 'figures', 'property_' + str(i) + '.png'), dpi=300)
            if show is True:
                plt.show()
            plt.clf()

    def calculate_ff_rmses(self):
        hvap_reference = []
        hmix_reference = []
        density_reference = []
        binary_density_reference = []
        hvap_rmse = []
        density_rmse = []
        hmix_rmse = []
        binary_density_rmse = []
        for property in self.properties.properties:
            if type(property) == enthalpy.EnthalpyOfVaporization:
                hvap_reference.append(property.value.m)
            elif type(property) == density.Density and property.substance.number_of_components == 1:
                density_reference.append(property.value.m)
            elif type(property) == density.Density and property.substance.number_of_components == 2:
                binary_density_reference.append(property.value.m)
            elif type(property) == enthalpy.EnthalpyOfMixing:
                hmix_reference.append(property.value.m)
        hvap_reference = np.asarray(hvap_reference)
        density_reference = np.asarray(density_reference)
        hmix_reference = np.asarray(hmix_reference)
        binary_density_reference = np.asarray(binary_density_reference)
        for i in range(self.property_measurements.values.shape[0]):
            hvap_estimate = []
            density_estimate = []
            hmix_estimate = []
            binary_density_estimate = []
            for j in range(len(self.property_measurements.values[0, :])):
                if self.property_measurements.columns[j].endswith('EnthalpyOfVaporization'):
                    hvap_estimate.append(self.property_measurements.values[i, j])
                elif self.property_measurements.columns[j].endswith('EnthalpyOfMixing'):
                    hmix_estimate.append(self.property_measurements.values[i, j])
                elif self.property_measurements.columns[j].endswith('Density') and '|' in \
                        self.property_measurements.columns[j]:
                    binary_density_estimate.append(self.property_measurements.values[i, j])
                elif self.property_measurements.columns[j].endswith('Density'):
                    density_estimate.append(self.property_measurements.values[i, j])
            hvap_estimate = np.asarray(hvap_estimate)
            density_estimate = np.asarray(density_estimate)
            if len(hvap_estimate) > 0:
                hvap_rmse.append(np.sqrt(np.mean(np.square(hvap_reference - hvap_estimate))))
            if len(density_estimate) > 0:
                density_rmse.append(np.sqrt(np.mean(np.square(density_reference - density_estimate))))
            if len(hmix_estimate) > 0:
                hmix_rmse.append(np.sqrt(np.mean(np.square(hmix_reference - hmix_estimate))))
            if len(binary_density_estimate) > 0:
                binary_density_rmse.append(
                    np.sqrt(np.mean(np.square(binary_density_reference - binary_density_estimate))))
        return hvap_rmse, density_rmse, hmix_rmse, binary_density_rmse

    def calculate_surrogate_simulation_deviation(self, benchmark_dataplex):
        params = benchmark_dataplex.parameter_values.values
        prediction_values = []
        prediction_uncertainties = []
        for i in range(params.shape[0]):
            with gpytorch.settings.eval_cg_tolerance(1e-2) and gpytorch.settings.fast_pred_samples(
                    True) and gpytorch.settings.fast_pred_var(True):
                eval = self.multisurrogate(torch.tensor(params[i, :]).unsqueeze(-1).T)
            prediction_values.append(eval.mean.detach().numpy())
            prediction_uncertainties.append(eval.variance.detach().numpy())
        prediction_values = np.asarray(prediction_values).squeeze(2)
        prediction_uncertainties = np.asarray(prediction_uncertainties).squeeze(2)
        deviation = abs(prediction_values - benchmark_dataplex.property_measurements.values)
        percent_deviation = 100 * deviation / benchmark_dataplex.property_measurements.values
        comb_uncert = prediction_uncertainties * 1.96 + benchmark_dataplex.property_uncertainties.values
        in_uncert = deviation / comb_uncert < 1
        return deviation, percent_deviation, comb_uncert, in_uncert


def get_training_data_new(data, properties, parameters, device):
    data_list = []
    for datum in data:
        data_list.append(ParameterSetData(datum))
    print('Collecting and Preparing Data...')
    dataplex = ParameterSetDataMultiplex(data_list, properties, parameters, device)
    before = copy.deepcopy(len(dataplex.multi_data))
    dataplex.check_parameters()
    dataplex.check_properties()
    after = len(dataplex.multi_data)
    print(f'Removed {before - after} incomplete or errored datasets')
    dataplex.align_property_data()
    # dataplex.plot_parameter_sets()
    # dataplex.plot_properties()
    print(f'Proceeding to build surrogates with {len(dataplex.multi_data)} Datasets')
    # dataplex.property_measurements.drop(columns=['CC(=O)O{solv}{x=1.000000}_EnthalpyOfVaporization','CC(=O)O{solv}{x=1.000000}_Density'],inplace=True)
    # dataplex.property_uncertainties.drop(
    #     columns=['CC(=O)O{solv}{x=1.000000}_EnthalpyOfVaporization', 'CC(=O)O{solv}{x=1.000000}_Density'],inplace=True)
    # temp_properties = PhysicalPropertyDataSet()
    # for property in dataplex.properties.properties[1:19]:
    #     temp_properties.add_properties(property)
    # for property in dataplex.properties.properties[20:]:
    #     temp_properties.add_properties(property)
    # dataplex.properties = temp_properties
    # temp_labels = dataplex.property_labels[1:19]
    # temp_labels.extend(dataplex.property_labels[20:])
    # dataplex.property_labels=temp_labels
    dataplex.build_multisurrogates(do_cross_validation=True)
    # dataplex.build_surrogates(do_cross_validation=False)
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
    for i, property in enumerate(dataset.properties):
        if property.id.startswith('000'):
            dataset.properties[i].id = property.id[3:]
        elif property.id.startswith('00'):
            dataset.properties[i].id = property.id[2:]
        elif property.id.startswith('0'):
            dataset.properties[i].id = property.id[1:]

        ids.append(int(property.id, 16))
    ids = sorted(ids)
    for i, id in enumerate(ids):
        ids[i] = format(id, 'x')
    properties = []
    property_ids = []
    for id in ids:
        for property in dataset.properties:
            if property.id == id:
                properties.append(property)
                property_ids.append(id)
    dataset._properties = tuple(properties)
    return dataset


def get_simulation_data(directory):
    from openff.evaluator.storage.storage import BaseStoredData
    data = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            data.append(BaseStoredData.from_json(os.path.join(directory, file)))
    return data


def calculate_ff_rmses_surrogate(dataplex, parameter_values):
    hvap_reference = []
    density_reference = []
    hvap_rmse = []
    density_rmse = []
    for property in dataplex.properties:
        if type(property) == enthalpy.EnthalpyOfVaporization:
            hvap_reference.append(property.value.m)
        elif type(property) == density.Density:
            density_reference.append(property.value.m)
    hvap_reference = np.asarray(hvap_reference)
    density_reference = np.asarray(density_reference)
    surrogates = []
    for i in range(parameter_values.shape[0]):
        param_vec = torch.tensor(parameter_values[i]).unsqueeze(-1).T
        surrogates.append(dataplex.multisurrogate(param_vec).mean.detach().numpy())
    surrogates = np.asarray(surrogates).squeeze()
    surrogates = pandas.DataFrame(surrogates, columns=dataplex.property_labels)

    for i in range(surrogates.values.shape[0]):
        hvap_estimate = []
        density_estimate = []
        for j in range(len(surrogates.values[0, :])):
            if surrogates.columns[j].endswith('EnthalpyOfVaporization'):
                hvap_estimate.append(surrogates.values[i, j])
            elif surrogates.columns[j].endswith('Density'):
                density_estimate.append(surrogates.values[i, j])
        hvap_estimate = np.asarray(hvap_estimate)
        density_estimate = np.asarray(density_estimate)
        hvap_rmse.append(np.mean(np.sqrt(np.square(hvap_reference - hvap_estimate))))
        density_rmse.append(np.mean(np.sqrt(np.square(density_reference - density_estimate))))
    return hvap_rmse, density_rmse
