import torch
import torch.distributions
import gpytorch
import copy
import numpy as np


class ObjectiveFunction(torch.nn.Module):
    def __init__(self, surrogates, targets, initial_params):
        super(ObjectiveFunction, self).__init__()
        self.surrogates = surrogates
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
        # self.flat_parameters = torch.tensor(self.flat_parameters.reshape(self.flat_parameters.shape[0],1).transpose())
        # self.flat_parameters = torch.tensor(np.expand_dims(self.flat_parameters, axis=1).transpose()).to(
        #     device=self.device)

    def initialize_parameters(self):
        self.flatten_parameters()
        self.params = torch.nn.Parameter(self.flat_parameters)

    def evaluate_parameter_set(self, parameter_set):
        predictions_all = []
        uncertainties_all = []
        for surrogate in self.surrogates:
            mean, variance = self.evaluate_surrogate_explicit_params(surrogate, parameter_set)
            predictions_all.append(mean)
            uncertainties_all.append(variance)
        uncertainties_all = torch.cat(uncertainties_all)
        predictions_all = torch.cat(predictions_all)
        uncertainties_all = uncertainties_all.reshape(uncertainties_all.shape[0], 1).to(device=self.device)
        predictions_all = predictions_all.reshape(predictions_all.shape[0], 1).to(device=self.device)

        return predictions_all, uncertainties_all

    def evaluate_surrogate_explicit_params(self, surrogate, parameter_set):
        with gpytorch.settings.eval_cg_tolerance(1e-2) and gpytorch.settings.fast_pred_samples(
                True) and gpytorch.settings.fast_pred_var(True):
            eval = surrogate(parameter_set)
        return eval.mean, eval.variance


class UnconstrainedGaussianObjectiveFunction(ObjectiveFunction):
    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set(
            torch.tensor(parameters).unsqueeze(-1).T)
        comb_uncert = torch.sqrt(torch.square(self.uncertainty_values) + torch.square(surrogate_uncertainties))

        objective = -torch.sum(
            torch.distributions.Normal(loc=self.experimental_values, scale=comb_uncert).log_prob(surrogate_predictions))

        return objective.item()


class ConstrainedGaussianObjectiveFunction(ObjectiveFunction):
    def __init__(self, surrogates, targets, initial_params, prior_width):
        super().__init__(surrogates, targets, initial_params)
        self.prior_width = prior_width

    def forward(self, parameters):
        surrogate_predictions, surrogate_uncertainties = self.evaluate_parameter_set(
            torch.tensor(parameters).unsqueeze(-1).T)
        comb_uncert = torch.sqrt(torch.square(self.uncertainty_values) + torch.square(surrogate_uncertainties))
        prior = torch.sum(torch.distributions.Normal(loc=torch.tensor(self.flat_parameters).unsqueeze(-1).T,
                                                     scale=self.prior_width * torch.tensor(self.flat_parameters).unsqueeze(
                                                         -1).T).log_prob(torch.tensor(parameters).unsqueeze(-1).T))
        objective = torch.sum(
            torch.distributions.Normal(loc=self.experimental_values, scale=comb_uncert).log_prob(surrogate_predictions))

        return -(prior.item() + objective.item())


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
