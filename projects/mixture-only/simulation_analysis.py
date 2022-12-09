import numpy as np
from LJ_surrogates.analysis.analysis import SimulatedDataset, unzip_rmse_dict, unzip_mse_dict
import matplotlib.pyplot as plt
import pandas as pd


dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-20/test-set-collection.json'
result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-20/estimated_results/estimated_data_set_32.json'

dataset_1 = SimulatedDataset(dataset_collection, result_dataset)

dataset_1.build_dataframe()
dataset_1.split_property_dataframes()
rmse_1 = dataset_1.calculate_rmses()
mse_1 = dataset_1.calculate_mses()
eom_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.enthalpy_of_mixing)
eov_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.enthalpy_of_vaporization)
pur_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.pure_density)
bin_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.binary_density)
eom_mses_1 = dataset_1.calculate_mse_by_category(dataset_1.enthalpy_of_mixing)
eov_mses_1 = dataset_1.calculate_mse_by_category(dataset_1.enthalpy_of_vaporization)
pur_mses_1 = dataset_1.calculate_mse_by_category(dataset_1.pure_density)
bin_mses_1 = dataset_1.calculate_mse_by_category(dataset_1.binary_density)
dataset_1.plot_all_parity()
print(rmse_1)
print(mse_1)


result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-10/estimated_results/estimated_data_set_17.json'
dataset_2 = SimulatedDataset(dataset_collection, result_dataset)
dataset_2.build_dataframe()
dataset_2.split_property_dataframes()
rmse = dataset_2.calculate_rmses()
mse = dataset_2.calculate_mses()
eom_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.enthalpy_of_mixing)
eov_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.enthalpy_of_vaporization)
pur_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.pure_density)
bin_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.binary_density)
eom_mses_2 = dataset_2.calculate_mse_by_category(dataset_2.enthalpy_of_mixing)
eov_mses_2 = dataset_2.calculate_mse_by_category(dataset_2.enthalpy_of_vaporization)
pur_mses_2 = dataset_2.calculate_mse_by_category(dataset_2.pure_density)
bin_mses_2 = dataset_2.calculate_mse_by_category(dataset_2.binary_density)
print(rmse)
print(mse)

dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-10/test-set-collection.json'
result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-10/estimated_results/estimated_data_set_10.json'
dataset_parsley = SimulatedDataset(dataset_collection, result_dataset)
dataset_parsley.build_dataframe()
dataset_parsley.split_property_dataframes()
rmse = dataset_parsley.calculate_rmses()
mse = dataset_parsley.calculate_mses()
eom_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.enthalpy_of_mixing)
eov_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.enthalpy_of_vaporization)
pur_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.pure_density)
bin_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.binary_density)
eom_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.enthalpy_of_mixing)
eov_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.enthalpy_of_vaporization)
pur_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.pure_density)
bin_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.binary_density)
print(rmse)
print(mse)

dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-10/test-set-collection.json'
result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/mixture-only-sim/training-set-eval/estimated_results/estimated_data_set_1.json'
dataset_sim = SimulatedDataset(dataset_collection, result_dataset)
dataset_sim.build_dataframe()
dataset_sim.split_property_dataframes()
rmse = dataset_sim.calculate_rmses()
mse = dataset_sim.calculate_mses()
eom_rmses_sim = dataset_sim.calculate_rmse_by_category(dataset_sim.enthalpy_of_mixing)
eov_rmses_sim = dataset_sim.calculate_rmse_by_category(dataset_sim.enthalpy_of_vaporization)
pur_rmses_sim = dataset_sim.calculate_rmse_by_category(dataset_sim.pure_density)
bin_rmses_sim = dataset_sim.calculate_rmse_by_category(dataset_sim.binary_density)
eom_mses_sim = dataset_sim.calculate_mse_by_category(dataset_sim.enthalpy_of_mixing)
eov_mses_sim = dataset_sim.calculate_mse_by_category(dataset_sim.enthalpy_of_vaporization)
pur_mses_sim = dataset_sim.calculate_mse_by_category(dataset_sim.pure_density)
bin_mses_sim = dataset_sim.calculate_mse_by_category(dataset_sim.binary_density)
print(rmse)
print(mse)

def build_improvement_plot(hmix_rmses_orig, hmix_rmses_new, density_rmses_orig, density_rmses_new, label, title, color):
    hmix_value_orig, hmix_labels_orig, hmix_errbar_orig = unzip_rmse_dict(hmix_rmses_orig)
    hmix_value_new, hmix_labels_new, hmix_errbar_new = unzip_rmse_dict(hmix_rmses_new)
    # hmix_value_sim, hmix_labels_sim, hmix_errbar_sim = unzip_rmse_dict(hmix_rmses_sim)
    density_value_orig, density_labels_orig, density_errbar_orig = unzip_rmse_dict(density_rmses_orig)
    density_value_new, density_labels_new, density_errbar_new = unzip_rmse_dict(density_rmses_new)
    # density_value_sim, density_labels_sim, density_errbar_sim = unzip_rmse_dict(density_rmses_sim)
    fig, ax = plt.subplots(1,2, figsize=(8,4))

    counter = np.arange(len(hmix_labels_new))+1
    ax[0].errorbar(hmix_value_orig, counter+0.25, xerr=hmix_errbar_orig, capsize=5, marker='o', alpha=1, ls='none', label='OpenFF 1.0.0', color='0.5')
    ax[0].errorbar(hmix_value_new, counter, xerr=hmix_errbar_new, capsize=5, marker='o', alpha=1, ls='none', label='multifidelity', color=color)
    ax[0].legend()
    ax[0].set_xlabel('RMSE, kJ/mol', fontsize=14)
    ax[0].set_yticks(counter)
    ax[0].set_yticklabels(hmix_labels_new)
    ax[0].set_xlim([0,1.25])
    ax[0].set_title(r'Training set $\Delta H_{mix}$')
    ax[1].errorbar(density_value_orig, counter+0.25, xerr=density_errbar_orig, capsize=5, marker='o', alpha=1, ls='none', label='OpenFF 1.0.0', color='0.5')
    ax[1].errorbar(density_value_new, counter, xerr=density_errbar_new, capsize=5, marker='o', alpha=1, ls='none', label='multifidelity', color=color)
    ax[1].legend()
    ax[1].set_xlabel('RMSE, g/ml', fontsize=14)
    ax[1].set_yticks([])
    ax[1].set_xlim([0,0.05])
    ax[1].set_title(r'Training set $\rho_L(x)$')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(label+'.png', dpi=300)


def build_improvement_plot_mse(hmix_mses_orig, hmix_mses_new, density_mses_orig, density_mses_new, label, title, color):
    hmix_value_orig, hmix_labels_orig, hmix_errbar_orig = unzip_mse_dict(hmix_mses_orig)
    hmix_value_new, hmix_labels_new, hmix_errbar_new = unzip_mse_dict(hmix_mses_new)
    density_value_orig, density_labels_orig, density_errbar_orig = unzip_mse_dict(density_mses_orig)
    density_value_new, density_labels_new, density_errbar_new = unzip_mse_dict(density_mses_new)

    fig, ax = plt.subplots(1,2, figsize=(8,4))

    counter = np.arange(len(hmix_labels_new))+1

    ax[0].errorbar(hmix_value_orig, counter+0.25, xerr=hmix_errbar_orig, capsize=5, marker='o', alpha=1, ls='none', label='OpenFF 1.0.0', color='0.5')
    ax[0].errorbar(hmix_value_new, counter, xerr=hmix_errbar_new, capsize=5, marker='o', alpha=1, ls='none', label='multifidelity', color=color)

    ax[0].legend()
    ax[0].set_xlabel('MSE, kJ/mol', fontsize=14)
    ax[0].set_yticks(counter)
    ax[0].set_yticklabels(hmix_labels_new)
    ax[0].set_xlim([-1.5,1])
    ax[0].axvline(0,color='k',ls='--')
    ax[0].set_title(r'Training set $\Delta H_{mix}$', fontsize=14)
    ax[1].errorbar(density_value_orig, counter+0.25, xerr=density_errbar_orig, capsize=5, marker='o', alpha=1, ls='none', label='OpenFF 1.0.0', color='0.5')
    ax[1].errorbar(density_value_new, counter, xerr=density_errbar_new, capsize=5, marker='o', alpha=1, ls='none', label='multifidelity', color=color)
    ax[1].legend()
    ax[1].set_xlabel('MSE, g/ml', fontsize=14)
    ax[1].set_yticks([])
    ax[1].axvline(0, color='k', ls='--')
    ax[1].set_xlim([-0.02,0.04])
    ax[1].set_title(r'Training set $\rho_L(x)$', fontsize=14)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(label+'.png', dpi=300)


build_improvement_plot(eom_rmses_parsley, eom_rmses_1, bin_rmses_parsley, bin_rmses_1, 'n=20_moieties', 'N=20', color='tab:orange')
build_improvement_plot(eom_rmses_parsley, eom_rmses_2, bin_rmses_parsley, bin_rmses_2, 'n=10_moieties', 'N=10', color='tab:blue')
build_improvement_plot_mse(eom_mses_parsley, eom_mses_1, bin_mses_parsley, bin_mses_1, 'n=20_moieties_mse', 'N=20', color='tab:orange')
build_improvement_plot_mse(eom_mses_parsley, eom_mses_2, bin_mses_parsley, bin_mses_2, 'n=10_moieties_mse', 'N=10', color='tab:blue')


dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/benchmark-set-collection.json'
result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_3/benchmark/estimated_results/estimated_data_set_1.json'

dataset_3 = SimulatedDataset(dataset_collection, result_dataset)
dataset_3.build_dataframe()
dataset_3.split_property_dataframes()
rmse = dataset_3.calculate_rmses()
mse = dataset_3.calculate_mses()
eom_rmses_3 = dataset_3.calculate_rmse_by_category(dataset_3.enthalpy_of_mixing)
eov_rmses_3 = dataset_3.calculate_rmse_by_category(dataset_3.enthalpy_of_vaporization)
pur_rmses_3 = dataset_3.calculate_rmse_by_category(dataset_3.pure_density)
bin_rmses_3 = dataset_3.calculate_rmse_by_category(dataset_3.binary_density)
eom_mses_3 = dataset_3.calculate_mse_by_category(dataset_3.enthalpy_of_mixing)
eov_mses_3 = dataset_3.calculate_mse_by_category(dataset_3.enthalpy_of_vaporization)
pur_mses_3 = dataset_3.calculate_mse_by_category(dataset_3.pure_density)
bin_mses_3 = dataset_3.calculate_mse_by_category(dataset_3.binary_density)

result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_4/benchmark/estimated_results/estimated_data_set_1.json'

dataset_4 = SimulatedDataset(dataset_collection, result_dataset)
dataset_4.build_dataframe()
dataset_4.split_property_dataframes()
rmse = dataset_4.calculate_rmses()
mse = dataset_4.calculate_mses()
eom_rmses_4 = dataset_4.calculate_rmse_by_category(dataset_4.enthalpy_of_mixing)
eov_rmses_4 = dataset_4.calculate_rmse_by_category(dataset_4.enthalpy_of_vaporization)
pur_rmses_4 = dataset_4.calculate_rmse_by_category(dataset_4.pure_density)
bin_rmses_4 = dataset_4.calculate_rmse_by_category(dataset_4.binary_density)
eom_mses_4 = dataset_4.calculate_mse_by_category(dataset_4.enthalpy_of_mixing)
eov_mses_4 = dataset_4.calculate_mse_by_category(dataset_4.enthalpy_of_vaporization)
pur_mses_4 = dataset_4.calculate_mse_by_category(dataset_4.pure_density)
bin_mses_4 = dataset_4.calculate_mse_by_category(dataset_4.binary_density)



result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/openff-1-0-0-benchmark/estimated_data_set_1.json'
dataset_parsley = SimulatedDataset(dataset_collection, result_dataset)
dataset_parsley.build_dataframe()
dataset_parsley.split_property_dataframes()
rmse = dataset_parsley.calculate_rmses()
mse = dataset_parsley.calculate_mses()
eom_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.enthalpy_of_mixing)
eov_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.enthalpy_of_vaporization)
pur_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.pure_density)
bin_rmses_parsley = dataset_parsley.calculate_rmse_by_category(dataset_parsley.binary_density)
eom_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.enthalpy_of_mixing)
eov_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.enthalpy_of_vaporization)
pur_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.pure_density)
bin_mses_parsley = dataset_parsley.calculate_mse_by_category(dataset_parsley.binary_density)


# print('Parsley Hvap')
# print(eov_rmses_parsley)
# print(eov_mses_parsley)
# print('Run 3 Hvap')
# print(eov_rmses_3)
# print(eov_mses_3)
# print('Run 4 Hvap')
# print(eov_rmses_4)
# print(eov_mses_4)
#
# print('Parsley Hmix')
# print(eom_rmses_parsley)
# print(eom_mses_parsley)
# print('Run 3 Hmix')
# print(eom_rmses_3)
# print(eom_mses_3)
# print('Run 4 Hmix')
# print(eom_rmses_4)
# print(eom_mses_4)
#
# print('Parsley Pure Density')
# print(pur_rmses_parsley)
# print(pur_mses_parsley)
# print('Run 3 Pure Density')
# print(pur_rmses_3)
# print(pur_mses_3)
# print('Run 4 Pure Density')
# print(pur_rmses_4)
# print(pur_mses_4)
#
# print('Parsley Binary Density')
# print(bin_rmses_parsley)
# print(bin_mses_parsley)
# print('Run 3 Binary Density')
# print(bin_rmses_3)
# print(bin_mses_3)
# print('Run 4 Binary Density')
# print(bin_rmses_4)
# print(bin_mses_4)

build_improvement_plot(eom_rmses_parsley, eom_rmses_3, bin_rmses_parsley, bin_rmses_3, 'run_3', 'run 3')
build_improvement_plot(eom_rmses_parsley, eom_rmses_4, bin_rmses_parsley, bin_rmses_4, 'run_4', 'run 4')
build_improvement_plot_mse(eom_mses_parsley, eom_mses_3, bin_mses_parsley, bin_mses_3, 'run_3_mse', 'run 3')
build_improvement_plot_mse(eom_mses_parsley, eom_mses_4, bin_mses_parsley, bin_mses_4, 'run_4_mse', 'run 4')