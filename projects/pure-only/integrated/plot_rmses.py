from LJ_surrogates.analysis.analysis import SimulatedDataset, unzip_rmse_dict
import numpy as np
import matplotlib.pyplot as plt

dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_1/benchmark/optimized_ffs/1/test-set-collection.json'
result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_1/benchmark/estimated_results/estimated_data_set_1.json'

dataset_1 = SimulatedDataset(dataset_collection, result_dataset)

dataset_1.build_dataframe()
dataset_1.split_property_dataframes()
rmse_1 = dataset_1.calculate_rmses()
eom_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.enthalpy_of_mixing)
eov_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.enthalpy_of_vaporization)
pur_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.pure_density)
bin_rmses_1 = dataset_1.calculate_rmse_by_category(dataset_1.binary_density)
dataset_1.plot_all_parity()

result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/openff-1-0-0-benchmark/estimated_data_set_1.json'
dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_1/benchmark/optimized_ffs/1/test-set-collection.json'

dataset_2 = SimulatedDataset(dataset_collection, result_dataset)

dataset_2.build_dataframe()
dataset_2.split_property_dataframes()
rmse_2 = dataset_2.calculate_rmses()
eom_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.enthalpy_of_mixing)
eov_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.enthalpy_of_vaporization)
pur_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.pure_density)
bin_rmses_2 = dataset_2.calculate_rmse_by_category(dataset_2.binary_density)
dataset_2.plot_all_parity()

eom_sim_only = {'RMSE': 7.47, 'CI': (5.47, 9.45)}
eov_sim_only = {'RMSE': 0.436, 'CI': (0.395, 0.478)}
pur_sim_only = {'RMSE': 0.0179, 'CI': (0.0125, 0.0228)}
bin_sim_only = {'RMSE': 0.0176, 'CI': (0.0156, 0.0198)}


rmse_simulation_only = {'Enthalpy of Vaporization': eov_sim_only, 'Enthalpy of Mixing': eom_sim_only, 'Pure Density': pur_sim_only, 'Binary Density': bin_sim_only}

rmses_sim_only, labels, errbar_sim_only = unzip_rmse_dict(rmse_simulation_only)
rmses_1, labels, errbar_1 = unzip_rmse_dict(rmse_1)
rmses_orig, labels, errbar_orig = unzip_rmse_dict(rmse_1)

rmses_all = np.vstack(rmses_orig,rmses_sim_only, rmses_1)

errbar_all = [errbar_orig, errbar_sim_only, errbar_1]

fig, ax = plt.subplots(1,4,figsize=(15,4))
for i in range(4):
    ax[i].bar(np.arange(3))