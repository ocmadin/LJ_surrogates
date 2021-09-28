import seaborn
import matplotlib.pyplot as plt
import pandas
import textwrap
import os


def plot_sampling_boundaries_1D(x_values, ranges, **kwargs):
    plt.axvline(ranges, ls='--', color='k', lw=1)
    plt.axvline(max(x_values), ls='--', color='k', lw=1)


def plot_triangle(params, likelihood, ranges):
    df = pandas.DataFrame(params, columns=likelihood.flat_parameter_names)
    wrapper = textwrap.TextWrapper(width=25)
    columns = {}
    for i, column in enumerate(df.columns):
        columns[column] = wrapper.fill(column)
    df.rename(columns=columns, inplace=True)
    pairplot = seaborn.pairplot(df, kind='kde', corner=True)
    for i in range(pairplot.axes.shape[0]):
        for j in range(pairplot.axes.shape[0]):
            if i == j:
                pairplot.axes[i][j].axvline(ranges[i, 0], ls='--', color='k')
                pairplot.axes[i][j].axvline(ranges[i, 1], ls='--', color='k')
    plt.tight_layout()
    pairplot.savefig(os.path.join('result/figures', 'trace_with_sampling_boundaries.png'), dpi=300)
    plt.close()