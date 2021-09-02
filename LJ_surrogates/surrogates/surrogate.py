import numpy as np
import gpytorch
import torch
import gpflow
import jax.numpy as jnp

class GPSurrogateModel:
    def __init__(self,parameter_data,property_data):
        self.cuda = torch.device('cuda')
        self.X = torch.tensor(parameter_data).to(device=self.cuda)
        self.Y = torch.tensor(property_data.flatten()).to(device=self.cuda)
        # self.X_gpflow = parameter_data
        # self.Y_gpflow = property_data
        # self.X = torch.tensor(parameter_data)
        # self.Y = torch.tensor(property_data.flatten())

    def build_surrogate_GPytorch(self):
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.model = ExactGPModel(self.X, self.Y, self.likelihood).cuda()
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.model = ExactGPModel(self.X, self.Y, self.likelihood)
        self.model.train()
        self.likelihood.train()
        training_iter = 1000
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

    def build_sparse_surrogate_gpytorch(self):
        from gpytorch.means import ConstantMean
        from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
        from gpytorch.distributions import MultivariateNormal

        class GPRegressionModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.base_covar_module = ScaleKernel(RBFKernel())
                self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :],
                                                        likelihood=likelihood)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.model = GPRegressionModel(self.X, self.Y, self.likelihood).cuda()
        training_iter = 1000
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

    def build_surrogate_gpflow(self):
        kernel = gpflow.kernels.SquaredExponential()
        self.model_gpflow = gpflow.models.GPR(data=(self.X_gpflow,self.Y_gpflow), kernel=kernel)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.model_gpflow.training_loss, self.model_gpflow.trainable_variables,options=dict(maxiter=100))


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
