import os

import numpy
import numpy as np
import gpytorch
import torch
import gpflow
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas
import copy


class GPSurrogateModel:
    def __init__(self, parameter_data, property_data, device):
        self.device = torch.device('device')
        self.X = torch.tensor(parameter_data).to(device=self.device)
        self.Y = torch.tensor(property_data.flatten()).to(device=self.device)
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
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.model = GPRegressionModel(self.X, self.Y, self.likelihood)
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
        if self.device == 'cpu':
            self.model.cpu()
            self.likelihood.cpu()
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


def build_surrogate_lightweight(parameter_data, property_data, property_uncertainties, device):
    cuda = torch.device('cuda')
    X = torch.tensor(parameter_data).to(device=cuda)
    Y = torch.tensor(property_data.flatten()).to(device=cuda)
    Y_err = torch.tensor(property_uncertainties.flatten()).to(device=cuda)

    # X = torch.tensor(parameter_data)
    # Y = torch.tensor(property_data.flatten())
    # Y_err = torch.tensor(property_uncertainties.flatten())

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
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err))

    model = ExactGPModel(X, Y, likelihood).cuda()
    # model = ExactGPModel(X, Y, likelihood)

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
    if device == 'cpu':
        model.cpu()
        likelihood.cpu()
    # print(model.covar_module.base_kernel.lengthscale.detach().numpy())
    return model

def build_surrogate_lightweight_botorch(parameter_data, property_data, property_uncertainties, device):
    cuda = torch.device('cuda')
    X = torch.tensor(parameter_data).to(device=cuda)
    Y = torch.tensor(property_data.flatten()).unsqueeze(-1).to(device=cuda)
    Y_err = torch.tensor(property_uncertainties.flatten()).unsqueeze(-1).to(device=cuda)

    mean_module = gpytorch.means.ConstantMean()
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]))

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err)).cuda()
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err))
    from botorch.models import FixedNoiseGP
    model = FixedNoiseGP(X, Y, Y_err, covar_module=covar_module).cuda()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    from botorch.fit import fit_gpytorch_model
    fit_gpytorch_model(mll)
    model.eval()
    likelihood.eval()
    if device == 'cpu':
        model.cpu()
        likelihood.cpu()
    # print(model.covar_module.base_kernel.lengthscale.detach().numpy())
    return model

def build_multisurrogate_lightweight_botorch(parameter_data, property_data, property_uncertainties, device):
    cuda = torch.device('cuda')
    X = torch.tensor(parameter_data).to(device=cuda)
    Y = torch.tensor(property_data).T.to(device=cuda)
    Y_err = torch.tensor(property_uncertainties).T.to(device=cuda)

    mean_module = gpytorch.means.ConstantMean()
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]))

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err)).cuda()
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err))
    from botorch.models import FixedNoiseGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_model
    model = FixedNoiseGP(X, Y, Y_err, covar_module=covar_module).cuda()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    model.likelihood.eval()
    if device == 'cpu':
        model.cpu()
        likelihood.cpu()
    # print(model.covar_module.base_kernel.lengthscale.detach().numpy())
    return model

def build_multisurrogate_lightweight(parameter_data, property_data, property_uncertainties, device):
    cuda = torch.device('cuda')
    X = torch.tensor(parameter_data).to(device=cuda)

    # X = torch.tensor(parameter_data)
    # Y = torch.tensor(property_data.flatten())
    # Y_err = torch.tensor(property_uncertainties.flatten())
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

    num_surrogates = property_data.shape[0]
    models = []
    likelihoods = []
    for i in range(num_surrogates):
        individual_property_measurements = property_data[i].reshape(
            (property_data[0].shape[0], 1))
        Y = torch.tensor(individual_property_measurements.flatten()).to(device=cuda)
        individual_property_uncertainties = property_uncertainties[i].reshape(
            (property_uncertainties[0].shape[0], 1))
        Y_err = torch.tensor(individual_property_uncertainties.flatten()).to(device=cuda)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.square(Y_err)).cuda()
        likelihoods.append(likelihood)
        models.append(ExactGPModel(X, Y, likelihood).cuda())

    model = gpytorch.models.IndependentModelList(*models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)

    # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # self.model = ExactGPModel(self.X, self.Y, self.likelihood)
    model.train()
    likelihood.train()
    training_iter = 1000
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(*model.train_inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    if device == 'cpu':
        model.cpu()
        likelihood.cpu()
    # print(model.covar_module.base_kernel.lengthscale.detach().numpy())
    return model



def build_surrogates_loo_cv(parameter_data, property_data, property_uncertainties, labels):
    cuda = torch.device('cuda')
    X = torch.tensor(parameter_data).to(device=cuda)
    Y = torch.tensor(property_data).T.to(device=cuda)
    Y_err = torch.tensor(property_uncertainties).T.to(device=cuda)
    # X = torch.tensor(parameter_data)
    # Y = torch.tensor(property_data.flatten())
    # Y_err = torch.tensor(property_uncertainties.flatten())
    from botorch.cross_validation import gen_loo_cv_folds
    cv_folds = gen_loo_cv_folds(X, Y, Y_err)
    train_X = cv_folds.train_X
    train_Y = cv_folds.train_Y
    train_Y_err = cv_folds.train_Yvar
    test_X = cv_folds.test_X
    test_Y = cv_folds.test_Y.squeeze(1).cpu().numpy()
    test_Y_err = cv_folds.test_Yvar.squeeze(1).cpu().numpy()

    from botorch.models import FixedNoiseGP
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]))
    means = []
    uncertainties = []
    for i in range(train_X.shape[0]):
        model = FixedNoiseGP(train_X[i], train_Y[i], train_Y_err[i],covar_module=covar_module).cuda()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        from botorch.fit import fit_gpytorch_model
        fit_gpytorch_model(mll)

        model.eval()
        model.likelihood.eval()

        output = model(test_X[i])
        mean = output.mean.detach().cpu().numpy()
        stdev = output.stddev.detach().cpu().numpy()
        means.append(mean)
        uncertainties.append(stdev)
    means = np.asarray(means)
    uncertainties = np.asarray(uncertainties)
    if property_data.shape[0] > 1:
        means = means.squeeze()
        uncertainties = uncertainties.squeeze()
    os.makedirs('validation', exist_ok=True)
    for i in range(len(labels)):
        df = pandas.DataFrame(
            np.vstack((test_Y[:,i], test_Y_err[:,i], means[:,i], uncertainties[:,i])).T,
            columns=['Simulated Value', 'Simulated Uncertainty', 'Surrogate Value', 'Surrogate Uncertainty'])
        df.to_csv(os.path.join('validation', 'cross_validation_' + str(labels[i]) + '.csv'))

        xax = [min(means[:,i]) * 0.9, max(means[:,i]) * 1.1]
        yax = [min(means[:,i]) * 0.9, max(means[:,i]) * 1.1]
        RMSE = np.sqrt(np.mean(np.square(means[:,i] - test_Y[:,i])))
        avg_surrogate_uncertainty = np.mean(uncertainties[:,i])
        print(f'LOO Cross-Validation for {labels[i]} surrogate:')
        print(f'Surrogate RMSE from Simulation: {RMSE}')
        print(f'Max Surrogate Error from Simulation: {max(means[:,i] - test_Y[:,i])}')
        print(f'Average Surrogate Uncertainty: {avg_surrogate_uncertainty}')
        print(f'Max Surrogate Uncertainty:{max(uncertainties[:,i])}')
        plt.errorbar(means[:,i], test_Y[:,i], xerr=1.96 * uncertainties[:,i], yerr=test_Y_err[:,i], ls='none',
                     marker='.')
        plt.plot(xax, yax, color='k', lw=0.5)
        plt.title(f'LOO Cross-validation for surrogate \n {labels[i]}')
        plt.xlabel('Surrogate Output')
        plt.ylabel('Simulation Value')
        plt.savefig(os.path.join('validation', 'cross-validation_' + str(labels[i]) + '.png'))
        plt.close()


def compute_surrogate_gradients(surrogate, point, eps, device):
    gradients = []
    for i, param in enumerate(point):
        perturbed_point_1 = copy.deepcopy(point)
        perturbed_point_1[i] = point[i] + eps * point[i]
        perturbed_point_2 = copy.deepcopy(point)
        perturbed_point_2[i] = point[i] - eps * point[i]
        gradient = (surrogate(
            torch.tensor(numpy.expand_dims(perturbed_point_1, axis=1).T).to(device=device)).mean - surrogate(
            torch.tensor(numpy.expand_dims(perturbed_point_2, axis=1).T).to(device=device)).mean) / (2 * eps * point[i])

        gradients.append(gradient)
    return gradients
