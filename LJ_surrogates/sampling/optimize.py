import os

import pandas
import torch
import torch.distributions
import gpytorch
import copy
import numpy as np
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from simtk import openmm, unit
import time


class ObjectiveFunction(torch.nn.Module):
    def __init__(self, multisurrogate, targets, initial_params):
        super(ObjectiveFunction, self).__init__()
        self.multisurrogate = multisurrogate
        self.targets = targets
        self.params = initial_params
        self.device = 'cpu'
        experiment_vector = []
        uncertainty_vector = []
        for property in self.targets.properties:
            experiment_vector.append(property.value.m)
            uncertainty_vector.append(property.uncertainty.m)
        self.experimental_values = torch.tensor(experiment_vector).unsqueeze(-1).to(device=self.device)
        self.uncertainty_values = torch.tensor(uncertainty_vector).unsqueeze(-1).to(device=self.device)

    def flatten_parameters(self):
        self.flat_parameters = []
        self.flat_parameter_names = []
        for key in self.params.keys():
            self.flat_parameters.append(self.params[key][0]._value)
            self.flat_parameters.append(self.params[key][1]._value)
            self.flat_parameter_names.append(key + '_epsilon')
            self.flat_parameter_names.append(key + '_rmin_half')
        self.flat_parameters = np.asarray(self.flat_parameters)

    def initialize_parameters(self):
        self.flatten_parameters()
        self.params = torch.nn.Parameter(self.flat_parameters)

    # def evaluate_parameter_set(self, parameter_set):
    #     predictions_all = []
    #     uncertainties_all = []
    #     for surrogate in self.surrogates:
    #         mean, variance = self.evaluate_surrogate_explicit_params(surrogate, parameter_set)
    #         predictions_all.append(mean)
    #         uncertainties_all.append(variance)
    #     uncertainties_all = torch.cat(uncertainties_all)
    #     predictions_all = torch.cat(predictions_all)
    #     uncertainties_all = uncertainties_all.reshape(uncertainties_all.shape[0], 1).to(device=self.device)
    #     predictions_all = predictions_all.reshape(predictions_all.shape[0], 1).to(device=self.device)
    #
    #     return predictions_all, uncertainties_all
    #
    def evaluate_surrogate_explicit_params(self, surrogate, parameter_set):
        with gpytorch.settings.eval_cg_tolerance(1e-2) and gpytorch.settings.fast_pred_samples(
                True) and gpytorch.settings.fast_pred_var(True):
            eval = surrogate(parameter_set)
        return eval.mean, eval.variance

    def evaluate_parameter_set_multisurrogate(self, parameter_set):
        self.parameter_set = parameter_set
        mean, variance = self.evaluate_surrogate_explicit_params(self.multisurrogate, parameter_set)
        return mean, variance


class UnconstrainedGaussianObjectiveFunction(ObjectiveFunction):
    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)
        comb_uncert = torch.sqrt(torch.square(self.uncertainty_values) + torch.square(surrogate_uncertainties))

        objective = -torch.sum(
            torch.distributions.Normal(loc=self.experimental_values, scale=comb_uncert).log_prob(surrogate_predictions))

        return objective.item()


class ConstrainedGaussianObjectiveFunction(ObjectiveFunction):
    def __init__(self, multisurrogate, targets, initial_params, prior_width):
        super().__init__(multisurrogate, targets, initial_params)
        self.prior_width = prior_width

    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)
        comb_uncert = torch.sqrt(torch.square(self.uncertainty_values) + torch.square(surrogate_uncertainties))
        prior = torch.sum(torch.distributions.Normal(loc=torch.tensor(self.flat_parameters).unsqueeze(-1).T,
                                                     scale=self.prior_width * torch.tensor(
                                                         self.flat_parameters).unsqueeze(
                                                         -1).T).log_prob(torch.tensor(parameters).unsqueeze(-1).T))
        objective = torch.sum(
            torch.distributions.Normal(loc=self.experimental_values, scale=comb_uncert).log_prob(surrogate_predictions))

        return -(prior.item() + objective.item())


class ConstrainedGaussianObjectiveFunctionNoSurrogate(ObjectiveFunction):
    def __init__(self, multisurrogate, targets, initial_params, prior_width):
        super().__init__(multisurrogate, targets, initial_params)
        self.prior_width = prior_width

    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)
        prior = torch.sum(torch.distributions.Normal(loc=torch.tensor(self.flat_parameters).unsqueeze(-1).T,
                                                     scale=self.prior_width * torch.tensor(
                                                         self.flat_parameters).unsqueeze(
                                                         -1).T).log_prob(torch.tensor(parameters).unsqueeze(-1).T))
        objective = torch.sum(
            torch.distributions.Normal(loc=self.experimental_values, scale=self.uncertainty_values).log_prob(
                surrogate_predictions))

        return -(prior.item() + objective.item())


class LeastSquaresObjectiveFunction(ObjectiveFunction):

    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)
        comb_uncert = torch.sqrt(torch.square(self.uncertainty_values) + torch.square(surrogate_uncertainties))
        objective = torch.square(surrogate_predictions - self.experimental_values) / self.experimental_values

        return objective.sum().item()


class ForceBalanceObjectiveFunction(ObjectiveFunction):
    def __init__(self, multisurrogate, targets, initial_params, property_labels):
        super().__init__(multisurrogate, targets, initial_params)
        self.property_labels = property_labels
        density_labels = []
        hvap_labels = []
        hmix_labels = []
        for i, label in enumerate(property_labels):
            if label.endswith('Density'):
                density_labels.append(i)
            elif label.endswith('EnthalpyOfVaporization'):
                hvap_labels.append(i)
            elif label.endswith('EnthalpyOfMixing'):
                hmix_labels.append(i)
        self.hvap_labels = np.asarray(hvap_labels)
        self.hmix_labels = np.asarray(hmix_labels)
        self.density_labels = np.asarray(density_labels)
        self.hvap_measurements = self.experimental_values[self.hvap_labels]
        self.hmix_measurements = self.experimental_values[self.hmix_labels]
        self.density_measurements = self.experimental_values[self.density_labels]
        self.hvap_denominator = 25.683
        self.density_denominator = 0.0482 * 2
        self.hmix_denominator = 1.594

    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)

        if len(self.density_measurements) > 0:
            surrogates_density = surrogate_predictions[self.density_labels]
            density_objective = (1 / surrogates_density.shape[0]) * torch.sum(torch.square(
                (self.density_measurements - surrogates_density) / self.density_denominator))
        else:
            density_objective = 0
        if len(self.hvap_measurements) > 0:
            surrogates_hvap = surrogate_predictions[self.hvap_labels]
            hvap_objective = (1 / surrogates_hvap.shape[0]) * torch.sum(torch.square(
                (self.hvap_measurements - surrogates_hvap) / self.hvap_denominator))
        else:
            hvap_objective = 0
        if len(self.hmix_measurements) > 0:
            surrogates_hmix = surrogate_predictions[self.hmix_labels]
            hmix_objective = (1 / surrogates_hmix.shape[0]) * torch.sum(torch.square(
                (self.hmix_measurements - surrogates_hmix) / self.hmix_denominator))
        else:
            hmix_objective = 0
        objective = density_objective + hvap_objective + hmix_objective
        return objective.item()

    def forward_jac(self, parameters):
        jacobian = torch.autograd.functional.jacobian(self.evaluate_parameter_set_multisurrogate,
                                                      torch.tensor(parameters).unsqueeze(-1).T)[0].squeeze()
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)

        if len(self.density_measurements) > 0:
            jacobian_density = jacobian[self.density_labels]
            surrogates_density = surrogate_predictions[self.density_labels]
            obj_density_jacobian = -2 / (surrogates_density.shape[0] * self.density_denominator) * torch.sum(
                ((self.density_measurements - surrogates_density) / self.density_denominator) * jacobian_density,
                axis=0)
        if len(self.hvap_measurements) > 0:
            jacobian_hvap = jacobian[self.hvap_labels]
            surrogates_hvap = surrogate_predictions[self.hvap_labels]
            obj_hvap_jacobian = -2 / (surrogates_hvap.shape[0] * self.hvap_denominator) * torch.sum(
                ((self.hvap_measurements - surrogates_hvap) / self.hvap_denominator) * jacobian_hvap, axis=0)
        if len(self.hmix_measurements) > 0:
            jacobian_hmix = jacobian[self.hmix_labels]
            surrogates_hmix = surrogate_predictions[self.hmix_labels]
            obj_hmix_jacobian = -2 / (surrogates_hmix.shape[0] * self.hmix_denominator) * torch.sum(
                ((self.hmix_measurements - surrogates_hmix) / self.hmix_denominator) * jacobian_hmix, axis=0)

        objective_jacobian = obj_density_jacobian + obj_hvap_jacobian + obj_hmix_jacobian
        return objective_jacobian.detach().numpy()

    def simulation_objective(self, simulation_outputs):

        if len(self.density_measurements) > 0:
            simulation_density = torch.tensor(simulation_outputs[self.density_labels]).unsqueeze(-1)
            density_objective = (1 / simulation_density.shape[0]) * torch.sum(torch.square(
                (self.density_measurements - simulation_density) / self.density_denominator))
        else:
            density_objective = 0
        if len(self.hvap_measurements) > 0:
            simulation_hvap = torch.tensor(simulation_outputs[self.hvap_labels]).unsqueeze(-1)
            hvap_objective = (1 / simulation_hvap.shape[0]) * torch.sum(torch.square(
                (self.hvap_measurements - simulation_hvap) / self.hvap_denominator))
        else:
            hvap_objective = 0
        if len(self.hmix_measurements) > 0:
            simulation_hmix = torch.tensor(simulation_outputs[self.hvap_labels]).unsqueeze(-1)
            hmix_objective = (1 / simulation_hmix.shape[0]) * torch.sum(torch.square(
                (self.hmix_measurements - simulation_hmix) / self.hmix_denominator))
        else:
            hmix_objective = 0
        objective = density_objective + hvap_objective + hmix_objective
        return objective.item()

    def surrogate_avg_uncertainty(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set_multisurrogate(
            torch.tensor(parameters).unsqueeze(-1).T)
        if len(self.density_measurements) > 0:
            surrogates_density = surrogate_predictions[self.density_labels]
            surrogates_density_uncertainty = surrogate_uncertainties[self.density_labels]
            avg_density_uncertainty = torch.mean(surrogates_density_uncertainty / surrogates_density)
        else:
            avg_density_uncertainty = None
        if len(self.hvap_measurements) > 0:
            surrogates_hvap = surrogate_predictions[self.hvap_labels]
            surrogates_hvap_uncertainty = surrogate_uncertainties[self.hvap_labels]
            avg_hvap_uncertainty = torch.mean(surrogates_hvap_uncertainty / surrogates_hvap)
        else:
            avg_hvap_uncertainty = None
        if len(self.hmix_measurements) > 0:
            surrogates_hmix = surrogate_predictions[self.hmix_labels]
            surrogates_hmix_uncertainty = surrogate_uncertainties[self.hmix_labels]
            avg_hmix_uncertainty = torch.mean(surrogates_hmix_uncertainty / surrogates_hmix)
        else:
            avg_hmix_uncertainty = None
        return avg_density_uncertainty, avg_hvap_uncertainty, avg_hmix_uncertainty


class Optimizer:
    def __init__(self, objective_function):
        self.objective_function = objective_function

    def training_loop(self, num_samples):
        losses = []
        parameters = []
        for i in range(num_samples):
            loss = self.objective_function(self.objective_function.params)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.detach().numpy())
            parameters.append(copy.deepcopy(self.objective_function.params.detach().numpy()))
        losses = np.asarray(losses)
        parameters = np.asarray(parameters).squeeze()
        return losses, parameters


class TorchADAMOptimizer(Optimizer):
    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.optimizer = torch.optim.Adam(objective_function.parameters(), lr=0.01)


class TorchAdagradOptimizer(Optimizer):
    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.optimizer = torch.optim.Adagrad(objective_function.parameters(), lr=0.01)


def create_forcefields_from_optimized_params(params, labels, input_forcefield):
    params = np.asarray(params)
    df = pandas.DataFrame(params, columns=labels)
    os.makedirs('optimized_ffs', exist_ok=True)
    for i in range(df.shape[0]):
        os.makedirs(os.path.join('optimized_ffs', str(i + 1)), exist_ok=True)
        forcefield = ForceField(input_forcefield)
        lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
        for j in range(df.shape[1]):
            smirks = df.columns[j].split('_')[0]
            param = df.columns[j].split('_')[1]
            for lj in lj_params:
                if lj.smirks == smirks:
                    if param == 'epsilon':
                        lj.epsilon = df.values[i, j] * unit.kilocalorie_per_mole
                    elif param == 'rmin':
                        lj.rmin_half = df.values[i, j] * unit.angstrom
        forcefield.to_file(os.path.join('optimized_ffs', str(i + 1), 'force-field.offxml'))
    forcefield = ForceField(input_forcefield)
    os.makedirs(os.path.join('optimized_ffs', str(df.shape[0] + 1)), exist_ok=True)
    forcefield.to_file(os.path.join('optimized_ffs', str(df.shape[0] + 1), 'force-field.offxml'))


def create_forcefield_for_simulation(params, labels, input_forcefield):
    params = np.asarray(params)
    df = pandas.DataFrame(params, columns=labels)
    os.makedirs('optimized_ffs', exist_ok=True)
    for i in range(df.shape[0]):
        os.makedirs(os.path.join('optimized_ffs', str(i + 1)), exist_ok=True)
        forcefield = ForceField(input_forcefield)
        lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
        for j in range(df.shape[1]):
            smirks = df.columns[j].split('_')[0]
            param = df.columns[j].split('_')[1]
            for lj in lj_params:
                if lj.smirks == smirks:
                    if param == 'epsilon':
                        lj.epsilon = df.values[i, j] * unit.kilocalorie_per_mole
                    elif param == 'rmin':
                        lj.rmin_half = df.values[i, j] * unit.angstrom
        forcefield.to_file(os.path.join('optimized_ffs', str(i + 1), 'force-field.offxml'))
    forcefield = ForceField(input_forcefield)
    os.makedirs(os.path.join('optimized_ffs', str(df.shape[0] + 1)), exist_ok=True)
    forcefield.to_file(os.path.join('optimized_ffs', str(df.shape[0] + 1), 'force-field.offxml'))
