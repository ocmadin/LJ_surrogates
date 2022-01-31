import numpy
import seaborn
import matplotlib.pyplot as plt
import pandas
import textwrap
import os
import scipy.stats.distributions
import numpy as np


def plot_sampling_boundaries_1D(x_values, ranges, **kwargs):
    plt.axvline(ranges, ls='--', color='k', lw=1)
    plt.axvline(max(x_values), ls='--', color='k', lw=1)


def plot_triangle(params, likelihood, ranges, opt_params, n, boundaries=True, maxima=False, show=False):
    df = pandas.DataFrame(params[:-1], columns=likelihood.flat_parameter_names)
    df2 = pandas.DataFrame(np.expand_dims(params[-1], axis=0), columns=likelihood.flat_parameter_names)
    wrapper = textwrap.TextWrapper(width=25)
    columns = {}
    for i, column in enumerate(df.columns):
        columns[column] = wrapper.fill(column)
    df.rename(columns=columns, inplace=True)
    pairplot = seaborn.pairplot(df.sample(n=n), kind='kde', corner=True)
    for i in range(pairplot.axes.shape[0]):
        for j in range(pairplot.axes.shape[0]):
            if i == j and boundaries is True:
                pairplot.axes[i][j].axvline(ranges[i, 0], ls='--', color='k')
                pairplot.axes[i][j].axvline(ranges[i, 1], ls='--', color='k')
                pairplot.axes[i][j].axvline(
                    scipy.stats.distributions.norm(df2.values[0][i], df2.values[0][i] / 10).ppf(0.025), ls='--',
                    color='r')
                pairplot.axes[i][j].axvline(
                    scipy.stats.distributions.norm(df2.values[0][i], df2.values[0][i] / 10).ppf(0.975), ls='--',
                    color='r')
            elif i > j and maxima is True:
                for param_set in opt_params:
                    pairplot.axes[i][j].scatter(param_set[j], param_set[i], marker='x', color='k')
    plt.tight_layout()
    if show is True:
        plt.show()
    pairplot.savefig(os.path.join('result/figures', 'trace_with_sampling_boundaries.png'), dpi=300)
    # pairplot.savefig('trace_with_opt.png', dpi=300)
    plt.close()


def plot_parameter_changes(optimized_params, original_params, parameter_labels, optimization_labels):
    plt.figure(figsize=(10, 10))
    percentages = []
    for i, param_set in enumerate(optimized_params):
        if i > 1:
            plt.plot(100 * param_set / original_params, label=optimization_labels[i], ls='-.', alpha=0.4, color='b')
        else:
            plt.plot(100 * param_set / original_params, label=optimization_labels[i])
        percentages.append(100 * param_set / original_params)
    plt.axhline(100, ls='--', color='k', label='Original force field')
    x = np.linspace(0, 11, 12)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.xticks(x, parameter_labels, rotation='vertical')
    plt.xlabel('Parameter')
    plt.ylabel('% Change from original force field')
    plt.title('Parameter Changes')
    plt.legend()
    plt.show()
