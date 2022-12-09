from LJ_surrogates.sampling.optimize import LeastSquaresObjectiveFunction, ForceBalanceObjectiveFunction, \
    create_forcefields_from_optimized_params, ConstrainedGaussianObjectiveFunction
from LJ_surrogates.surrogates.collate_data import collate_physical_property_data, calculate_ff_rmses_surrogate
import torch
import gc
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize, least_squares
import numpy as np
import pandas
import textwrap
import seaborn
import os
from LJ_surrogates.plotting.plotting import plot_triangle
from gpytorch.constraints import GreaterThan
import time

device = torch.device('cuda')
path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/pure-only-pooled/'
smirks_types_to_change = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
forcefield = 'openff-1.0.0.offxml'
dataset_json = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only/test-set-collection.json'
device = 'cpu'


pooled_dataplex = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                          dataset_json, device, constraint=None)

def get_random_parameter_vector(bounds):
    bounds_range = bounds[:, 1] - bounds[:, 0]
    rand = np.random.rand(bounds[:, 0].shape[0])
    vec = bounds[:, 0] + rand * bounds_range
    return vec


def compare_point(parameters, simulation_results, dataplex1, dataplex_ref):
    objective1 = ForceBalanceObjectiveFunction(dataplex1.multisurrogate, dataplex1.properties,
                                               dataplex1.initial_parameters,
                                               dataplex1.property_labels)
    objective1.flatten_parameters()

    objective_ref = ForceBalanceObjectiveFunction(dataplex_ref.multisurrogate, dataplex_ref.properties,
                                                  dataplex_ref.initial_parameters,
                                                  dataplex_ref.property_labels)
    objective_ref.flatten_parameters()

    obj_sim = objective1.simulation_objective(simulation_results)

    obj_eval_1 = objective1.forward(parameters)
    obj_eval_ref = objective_ref.forward(parameters)

    return np.asarray([obj_eval_1, obj_eval_ref, obj_sim])


path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_1/estimated_results'

dataplex_10_1 = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                               dataset_json, device, constraint=None)

path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_2/estimated_results'

dataplex_10_2 = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                               dataset_json, device, constraint=GreaterThan(1e-5))

path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_3/estimated_results'

dataplex_10_3 = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                               dataset_json, device, constraint=GreaterThan(1e-5))

path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_4/estimated_results'

dataplex_10_4 = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                               dataset_json, device, constraint=GreaterThan(1e-5))

path = '/home/owenmadin/storage/LINCOLN1/surrogate_modeling/pure-only-10/integrated_run_5/estimated_results'

dataplex_10_5 = collate_physical_property_data(path, smirks_types_to_change, forcefield,
                                               dataset_json, device, constraint=GreaterThan(1e-5))


points = [dataplex_10_1.parameter_values.values[24], dataplex_10_2.parameter_values.values[19], dataplex_10_3.parameter_values.values[23], dataplex_10_4.parameter_values.values[15], dataplex_10_5.parameter_values.values[16]]
measurements = [dataplex_10_1.property_measurements.values[24], dataplex_10_2.property_measurements.values[19], dataplex_10_3.property_measurements.values[23], dataplex_10_4.property_measurements.values[15], dataplex_10_5.property_measurements.values[16]]

result = []

dataplexes = [dataplex_10_1, dataplex_10_2, dataplex_10_3, dataplex_10_4, dataplex_10_5]

for dataplex in dataplexes:
    dataplex_result = []
    for i, point in enumerate(points):
        dataplex_result.append(compare_point(point, measurements[i], dataplex, pooled_dataplex))
    dataplex_result = np.asarray(dataplex_result)
    result.append((dataplex_result[:,0]-dataplex_result[:,2])*100/dataplex_result[:,2])

result = np.round(np.asarray(result),3)

# result[4, 2] = 0.200

labels = ['run 1', 'run 2', 'run 3', 'run 4', 'run 5']
labels_optima = ['run 1 \n optima', 'run 2 \n optima', 'run 3 \n optima', 'run 4 \n optima', 'run 5 \n optima']
cbarlabel = 'Deviation from simulation objective'
fig, ax = plt.subplots()
im = ax.imshow(result, cmap='YlOrRd')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels_optima)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.set_ylabel('Surrogate', fontsize=14)
ax.set_xlabel('Optimized Point', fontsize=14)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# result[4,2] = 0.482
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, str(round(result[i, j],1))+'%',
                       ha="center", va="center", color="k")

ax.set_title("Surrogate Evaluation of Simulation Optima", fontsize=14)
fig.tight_layout()
plt.show()

def run_lbfgsb_optimizations(dataplex, num_replicates):
    objective = ForceBalanceObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                              dataplex.property_labels)
    objective.flatten_parameters()

    bounds = []
    for column in dataplex.parameter_values.columns:
        minbound = min(dataplex.parameter_values[column].values)
        maxbound = max(dataplex.parameter_values[column].values)
        bounds.append((minbound, maxbound))
    bounds = np.asarray(bounds)
    bounds_increment = 1.1
    bounds[:, 0] /= (bounds_increment ** (1))
    bounds[:, 1] *= (bounds_increment ** (1))
    objectives = []
    solutions = []
    initial_solutions = []

    for i in range(num_replicates):
        before = time.time()
        vec = get_random_parameter_vector(bounds)
        initial_solutions.append(vec)
        result_l_bfgs_b = minimize(objective, x0=vec, jac=None, bounds=bounds,
                                   method='L-BFGS-B')
        objectives.append(result_l_bfgs_b.fun)
        solutions.append(result_l_bfgs_b.x)
        after = time.time()
        print(f'Optimization {i + 1} finished in {after - before} seconds')
    return np.asarray(objectives), np.asarray(solutions), np.asarray(initial_solutions)


def run_diffev_optimizations(dataplex, num_replicates):
    objective = ForceBalanceObjectiveFunction(dataplex.multisurrogate, dataplex.properties, dataplex.initial_parameters,
                                              dataplex.property_labels)
    objective.flatten_parameters()

    bounds = []
    for column in dataplex.parameter_values.columns:
        minbound = min(dataplex.parameter_values[column].values)
        maxbound = max(dataplex.parameter_values[column].values)
        bounds.append((minbound, maxbound))
    bounds = np.asarray(bounds)
    bounds_increment = 1.1
    bounds[:, 0] /= (bounds_increment ** (1))
    bounds[:, 1] *= (bounds_increment ** (1))
    objectives = []
    solutions = []
    initial_solutions = []

    for i in range(num_replicates):
        before = time.time()
        result_l_bfgs_b = differential_evolution(objective, bounds=bounds)
        objectives.append(result_l_bfgs_b.fun)
        solutions.append(result_l_bfgs_b.x)
        after = time.time()
        print(f'Optimization {i + 1} finished in {after - before} seconds')
    return np.asarray(objectives), np.asarray(solutions)


# os.makedirs('surrogate-testing', exist_ok=True)
#
# objectives_1, solutions_1, initial_solutions_1 = run_lbfgsb_optimizations(dataplex_10_1, num_replicates=100)
#
# np.save('surrogate-testing/objectives_1.npy', objectives_1)
# np.save('surrogate-testing/solutions_1.npy', solutions_1)
# np.save('surrogate-testing/initial_solutions_1.npy', initial_solutions_1)
#
# objectives_2, solutions_2, initial_solutions_2 = run_lbfgsb_optimizations(dataplex_10_2, num_replicates=100)
#
# np.save('surrogate-testing/objectives_2.npy', objectives_2)
# np.save('surrogate-testing/solutions_2.npy', solutions_2)
# np.save('surrogate-testing/initial_solutions_2.npy', initial_solutions_2)
#
# objectives_3, solutions_3, initial_solutions_3 = run_lbfgsb_optimizations(dataplex_10_3, num_replicates=100)
#
# np.save('surrogate-testing/objectives_3.npy', objectives_3)
# np.save('surrogate-testing/solutions_3.npy', solutions_3)
# np.save('surrogate-testing/initial_solutions_3.npy', initial_solutions_3)
#
# objectives_4, solutions_4, initial_solutions_4 = run_lbfgsb_optimizations(dataplex_10_4, num_replicates=100)
#
# objectives_diffev_4, solutions_diffev_4 = run_diffev_optimizations(dataplex_10_4, num_replicates=20)
#
# np.save('surrogate-testing/objectives_4.npy', objectives_4)
# np.save('surrogate-testing/solutions_4.npy', solutions_4)
# np.save('surrogate-testing/initial_solutions_4.npy', initial_solutions_4)
#
# np.save('surrogate-testing/objectives_diffev_4.npy', objectives_4)
# np.save('surrogate-testing/solutions_diffev_4.npy', solutions_4)
#
# objectives_5, solutions_5, initial_solutions_5 = run_lbfgsb_optimizations(dataplex_10_5, num_replicates=100)
#
# np.save('surrogate-testing/objectives_5.npy', objectives_5)
# np.save('surrogate-testing/solutions_5.npy', solutions_5)
# np.save('surrogate-testing/initial_solutions_5.npy', initial_solutions_5)
#
#
#
#
#
#
#
#


