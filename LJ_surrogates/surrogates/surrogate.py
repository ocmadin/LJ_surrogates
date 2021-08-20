import matplotlib.pyplot as plt
import numpy as np
import gpflow
from fffit.fffit.models import run_gpflow_scipy
import gpytorch
import torch
class GPSurrogateModel:
    def __init__(self,parameter_data,property_data,property):
        self.X = torch.tensor(parameter_data)
        self.Y = torch.tensor(property_data.flatten())
    def build_surrogate(self):
        self.model = run_gpflow_scipy(self.X, self.Y,
                         gpflow.kernels.Matern12(lengthscales=0.5*np.ones(self.X.shape[1])))

    def build_surrogate_GPytorch(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.X, self.Y, self.likelihood)
        self.model.train()
        self.likelihood.train()
        training_iter = 50
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.Y)
            loss.backward()
            optimizer.step()
        self.model.eval()
        self.likelihood.eval()
        delattr(self, 'X')
        delattr(self, 'Y')



    def evaluate_parameters(self,parameter_set):
        if parameter_set.shape[1] != self.X.shape[1]:
            raise ValueError(f'Parameter set has the wrong number of parameters for this model. \n The model has {self.X.shape[0]} parameters, and you supplied a set with {parameter_set.shape[0]} parameters')
        prediction = self.model.predict_y(parameter_set)
        predictions = np.asarray(prediction[0])
        uncertainties = np.asarray(prediction[1])
        return predictions, uncertainties

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # initialize likelihood and model

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
