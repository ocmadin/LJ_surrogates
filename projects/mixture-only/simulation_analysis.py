from LJ_surrogates.analysis.analysis import SimulatedDataset
import numpy as np
import matplotlib.pyplot as plt


dataset_collection = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/benchmark-set-collection.json'
result_dataset = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/mixture-only/individual_surrogate_benchmark_1/estimated_data_set_1.json'

dataset = SimulatedDataset(dataset_collection, result_dataset)

dataset.build_dataframe()
dataset.split_property_dataframes()
rmse = dataset.calculate_rmses()
eom_rmses = dataset.calculate_rmse_by_category(dataset.enthalpy_of_mixing)
eov_rmses = dataset.calculate_rmse_by_category(dataset.enthalpy_of_vaporization)
pur_rmses = dataset.calculate_rmse_by_category(dataset.pure_density)
bin_rmses = dataset.calculate_rmse_by_category(dataset.binary_density)
dataset.plot_all_parity()


openff_1_0_0_rmses = [0.03, 9.93, 0.025, 0.61]
mixture_only_rmses = [0.023, 9.95, 0.020, 0.27]
mf_rmses = [0.022, 4.75, 0.019, 0.38]

all_rmses = np.vstack([openff_1_0_0_rmses, mixture_only_rmses, mf_rmses]).T

fig,ax = plt.subplots(1,4,figsize=(16,4))
colors = ['b','orange','green']
labels = ['OpenFF 1.0.0', 'Simulation', 'Multi-Fidelity']
properties = ['Pure Density', 'Enthalpy of Vaporization', 'Binary Density', 'Binary Enthalpy of Mixing']
units = ['g/mL', 'kJ/mol','g/mL', 'kJ/mol']
for i in range(all_rmses.shape[0]):
    ax[i].bar(labels,all_rmses[i], color=colors)
    ax[i].set_ylabel(f'RMSE, {units[i]}')
    ax[i].set_title(properties[i])
fig.suptitle('Benchmark Set Performance')
plt.tight_layout()
fig.show()



