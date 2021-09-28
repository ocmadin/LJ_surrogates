import os

import numpy
import numpy as np
import gpytorch
import torch
import gpflow
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas


class GPSurrogateModel:
    def __init__(self, parameter_data, property_data):
        self.cuda = torch.device('cuda')
        self.X = torch.tensor(parameter_data).to(device=self.cuda)
        self.Y = torch.tensor(property_data.flatten()).to(device=self.cuda)
        self.X_gpflow = parameter_data
        self.Y_gpflow = property_data
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
        self.model_gpflow = gpflow.models.GPR(data=(self.X_gpflow, self.Y_gpflow), kernel=kernel)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.model_gpflow.training_loss, self.model_gpflow.trainable_variables,
                                options=dict(maxiter=10000))

    def evaluate_parameters(self, parameter_set):
        if parameter_set.shape[1] != self.X.shape[1]:
            raise ValueError(
                f'Parameter set has the wrong number of parameters for this model. \n The model has {self.X.shape[0]} parameters, and you supplied a set with {parameter_set.shape[0]} parameters')
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


def build_surrogate_lightweight(parameter_data, property_data, property_uncertainties):
    cuda = torch.device('cuda')
    X = torch.tensor(parameter_data).to(device=cuda)
    Y = torch.tensor(property_data.flatten()).to(device=cuda)
    Y_err = torch.tensor(property_uncertainties.flatten()).to(device=cuda)


    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(0.25)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err)).cuda()
    model = ExactGPModel(X, Y, likelihood).cuda()
    # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # self.model = ExactGPModel(self.X, self.Y, self.likelihood)
    model.train()
    likelihood.train()
    training_iter = 1000
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    print(model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
    return model


def build_surrogates_loo_cv(parameter_data, property_data, property_uncertainties, label):
    X = torch.tensor(parameter_data).cuda()
    Y = torch.tensor(property_data.flatten()).cuda()
    Y_err = torch.tensor(property_uncertainties.flatten()).cuda()
    from botorch.cross_validation import gen_loo_cv_folds
    cv_folds = gen_loo_cv_folds(X, Y, Y_err)
    train_X = cv_folds.train_X
    train_Y = cv_folds.train_Y.squeeze(2)
    train_Y_err = cv_folds.train_Yvar.squeeze(2)
    test_X = cv_folds.test_X
    test_Y = cv_folds.test_Y.squeeze(1).cpu().numpy()
    test_Y_err = cv_folds.test_Yvar.squeeze(1).cpu().numpy()

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_x.shape[0]]))
            # self.covar_module = gpytorch.kernels.ScaleKernel(
            #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_x.shape[0]]), ard_num_dims=train_x.shape[1]),
            #     batch_shape=torch.Size([train_x.shape[0]]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_x.shape[0]])),
                batch_shape=torch.Size([train_x.shape[0]]))
                # initialize likelihood and model

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(train_Y_err),batch_shape=torch.Size([train_Y_err.shape[0]]))
    model = ExactGPModel(train_X, train_Y, likelihood).cuda()
    # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # self.model = ExactGPModel(self.X, self.Y, self.likelihood)
    model.train()
    likelihood.train()
    training_iter = 1000
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_X)
        # Calc loss and backprop gradients
        loss = -mll(output, train_Y).sum()
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()

    output = model(test_X)
    mean = output.mean.detach().cpu().numpy()
    stdev = output.stddev.detach().cpu().numpy()
    os.makedirs('validation', exist_ok=True)
    df = pandas.DataFrame(np.vstack((test_Y.squeeze(), property_uncertainties.squeeze(), mean.squeeze(), stdev.squeeze())).T,
                     columns=['Simulated Value', 'Simulated Uncertainty', 'Surrogate Value', 'Surrogate Uncertainty'])
    df.to_csv(os.path.join('validation', 'cross_validation_' + str(label) + '.csv'))
    xax = [min(mean) * 0.9, max(mean) * 1.1]
    yax = [min(mean) * 0.9, max(mean) * 1.1]
    RMSE = np.sqrt(np.mean(np.square(mean-test_Y)))
    avg_surrogate_uncertainty = np.mean(stdev)
    print(f'LOO Cross-Validation for {label} surrogate:')
    print(f'Surrogate RMSE from Simulation: {RMSE}')
    print(f'Max Surrogate Error from Simulation: {max(mean-test_Y)}')
    print(f'Average Surrogate Uncertainty: {avg_surrogate_uncertainty}')
    print(f'Max Surrogate Uncertainty:{max(stdev)}')
    plt.errorbar(mean, test_Y, xerr=1.96 * stdev.squeeze(), yerr=property_uncertainties.squeeze(), ls='none',
                 marker='.')
    plt.plot(xax, yax, color='k', lw=0.5)
    plt.title(f'LOO Cross-validation for surrogate \n {label}')
    plt.xlabel('Surrogate Output')
    plt.ylabel('Simulation Value')
    plt.savefig(os.path.join('validation', 'cross-validation_' + str(label) + '.png'))
    plt.close()
