import numpy as np
import numpyro.distributions
import pyro
import pyro.distributions
import torch
import torch.distributions
from pyro.infer import MCMC, NUTS, HMC
# from numpyro.infer import MCMC, NUTS, HMC
import numpyro.distributions as npdist
# from jax.random import PRNGKey
import gpytorch
import functools
import time
import arviz


class likelihood_function:
    def __init__(self, dataplex):
        self.cuda = torch.device('cuda')
        self.surrogates = dataplex.surrogates
        self.experimental_properties = dataplex.properties
        self.parameters = dataplex.initial_parameters
        self.flatten_parameters()
        experiment_vector = []
        uncertainty_vector = []
        for property in self.experimental_properties.properties:
            experiment_vector.append(property.value.m)
            uncertainty_vector.append(property.uncertainty.m)
        self.experimental_values = torch.tensor(experiment_vector).unsqueeze(-1).to(device=self.cuda)
        # self.experimental_values = torch.tensor(experiment_vector).unsqueeze(-1)
        self.uncertainty_values = torch.tensor(uncertainty_vector).unsqueeze(-1).to(device=self.cuda)
        # self.uncertainty_values = torch.tensor(uncertainty_vector).unsqueeze(-1)

    def flatten_parameters(self):
        self.flat_parameters = []
        self.flat_parameter_names = []
        for key in self.parameters.keys():
            self.flat_parameters.append(self.parameters[key][0]._value)
            self.flat_parameters.append(self.parameters[key][1]._value)
            self.flat_parameter_names.append(key + '_epsilon')
            self.flat_parameter_names.append(key + '_rmin_half')
        self.flat_parameters = np.asarray(self.flat_parameters)
        # self.flat_parameters = torch.tensor(self.flat_parameters.reshape(self.flat_parameters.shape[0],1).transpose())
        self.flat_parameters = torch.tensor(np.expand_dims(self.flat_parameters, axis=1).transpose()).to(
            device=self.cuda)

    def evaluate_parameter_set(self, parameter_set):
        self.parameter_set = parameter_set
        predictions_all = []
        uncertainties_all = []
        for surrogate in self.surrogates:
            mean, variance = self.evaluate_surrogate_explicit_params(surrogate, parameter_set)
            predictions_all.append(mean)
            uncertainties_all.append(variance)
        uncertainties_all = torch.cat(uncertainties_all)
        predictions_all = torch.cat(predictions_all)
        uncertainties_all = uncertainties_all.reshape(uncertainties_all.shape[0], 1).to(device=self.cuda)
        predictions_all = predictions_all.reshape(predictions_all.shape[0], 1).to(device=self.cuda)
        # predictions_all = predictions_all.reshape(predictions_all.shape[0], 1)
        # uncertainties_all = uncertainties_all.reshape(uncertainties_all.shape[0], 1)

        return predictions_all, uncertainties_all

    def evaluate_parameter_set_map(self, parameter_set):
        self.parameter_set = parameter_set
        x_map = list(map(self.evaluate_surrogate, self.surrogates))
        predictions = torch.tensor(x_map).cuda()
        return predictions[:, 0].unsqueeze(-1), predictions[:, 1].unsqueeze(-1)

    def evaluate_surrogate(self, surrogate):
        with gpytorch.settings.eval_cg_tolerance(1e-2) and gpytorch.settings.fast_pred_samples(
                True) and gpytorch.settings.fast_pred_var(True):
            eval = surrogate[1](surrogate[0](self.parameter_set))
        return eval.mean, eval.variance

    def evaluate_surrogate_explicit_params(self, surrogate, parameter_set):
        self.parameter_set = parameter_set
        with gpytorch.settings.eval_cg_tolerance(1e-2) and gpytorch.settings.fast_pred_samples(
                True) and gpytorch.settings.fast_pred_var(True):
            eval = surrogate[1](surrogate[0](self.parameter_set))
        return eval.mean, eval.variance

    def evaluate_surrogate_gpflow(self, surrogate, parameter_set):
        before = time.time()
        mean, variance = surrogate.model_gpflow.predict_f(np.expand_dims(parameter_set[:, 0], axis=1).transpose())
        after = time.time()
        print(f'Mean: {mean}')
        print(f'Variance: {variance}')
        print(f'GPflow: {after - before} seconds')
        return

    def pyro_model(self):
        # Place priors on the virtual site charges increments and distance.
        parameters = pyro.sample(
            "parameters",
            pyro.distributions.Normal(
                # Use a normal distribution centered at one and with a sigma of 0.5
                # to stop the distance collapsing to 0 or growing too large.
                loc=self.flat_parameters,
                scale=self.flat_parameters * 0.1,
            )
        )

        predictions, predicted_uncertainties = self.evaluate_parameter_set(parameters)
        uncertainty = pyro.deterministic(
            "uncertainty", torch.sqrt(torch.square(self.uncertainty_values) + torch.square(predicted_uncertainties)))
        # uncertainty = pyro.deterministic(
        #     "uncertainty", self.uncertainty_values)

        return pyro.sample(
            "predicted_residuals",
            pyro.distributions.Normal(loc=predictions, scale=uncertainty),
            obs=self.experimental_values,
        ).cuda()

        # return pyro.sample(
        #     "predicted_residuals",
        #     pyro.distributions.Normal(loc=predictions, scale=uncertainty),
        #     obs=self.experimental_values,
        # )

    def sample(self, samples, step_size=0.001, max_tree_depth=10, num_chains=1):
        # Train the parameters and plot the sampled traces.
        nuts_kernel = NUTS(self.pyro_model, step_size=step_size, max_tree_depth=max_tree_depth)
        initial_params = {'parameters': torch.tile(self.flat_parameters, (num_chains, 1))}

        self.mcmc = MCMC(nuts_kernel, initial_params=initial_params, num_samples=samples,
                         warmup_steps=int(np.floor(samples / 5)), num_chains=num_chains, mp_context='spawn')
        # self.mcmc = MCMC(nuts_kernel, num_samples=samples, num_warmup=int(np.floor(samples/5)), num_chains=1)

        self.mcmc.run()
        self.mcmc.summary()
        self.samples = self.mcmc.get_samples()

        return self.mcmc
