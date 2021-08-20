import numpy as np
import pandas as pd
import arviz
import pyro
import pyro.distributions
import torch
import torch.distributions
from matplotlib import pyplot
from pyro.infer import MCMC, NUTS

class likelihood_function:
    def __init__(self, dataplex):
        self.surrogates = dataplex.surrogates
        self.experimental_properties = dataplex.properties
        self.parameters = dataplex.initial_parameters
        self.flatten_parameters()
        experiment_vector = []
        uncertainty_vector = []
        for property in self.experimental_properties.properties:
            experiment_vector.append(property.value.m)
            uncertainty_vector.append(property.uncertainty.m)
        self.experimental_values = torch.tensor(experiment_vector)
        self.experimental_values = self.experimental_values.reshape(self.experimental_values.shape[0],1)
        self.uncertainty_values = torch.tensor(uncertainty_vector)
        self.uncertainty_values = self.uncertainty_values.reshape(self.uncertainty_values.shape[0],1)

    def flatten_parameters(self):
        self.flat_parameters = []
        for key in self.parameters.keys():
            self.flat_parameters.append(self.parameters[key][0]._value)
            self.flat_parameters.append(self.parameters[key][1]._value)
        self.flat_parameters = np.asarray(self.flat_parameters)
        self.flat_parameters = torch.tensor(self.flat_parameters.reshape(self.flat_parameters.shape[0],1).transpose())

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



    def pyro_model(self):

        # Place priors on the virtual site charges increments and distance.
        parameters = pyro.sample(
            "parameters",
            pyro.distributions.Normal(
                # Use a normal distribution centered at one and with a sigma of 0.5
                # to stop the distance collapsing to 0 or growing too large.
                self.flat_parameters,
                self.flat_parameters * 0.25,
            ),
        )
        predictions, predicted_uncertainties = self.evaluate_parameter_set(parameters)
        uncertainty = pyro.deterministic(
            "uncertainty",torch.sqrt(self.uncertainty_values**2 + predicted_uncertainties**2))

        return pyro.sample(
            "predicted_residuals",
            pyro.distributions.Normal(loc=predictions, scale=uncertainty),
            obs=self.experimental_values,
        )


    def sample(self,samples):
        #Train the parameters and plot the sampled traces.
        nuts_kernel = NUTS(self.pyro_model)

        self.mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=int(np.floor(samples/5)), num_chains=1)
        self.mcmc.run()





