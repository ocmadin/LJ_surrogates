import shutil

import numpy
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet
import pandas
import numpy as np
import os
from LJ_surrogates.parameter_modification import vary_parameters_lhc
from openff.toolkit.typing.engines.smirnoff.forcefield import ForceField
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions
from openff.evaluator.client import EvaluatorClient, ConnectionOptions
from simtk import openmm, unit
from openff.evaluator.utils import setup_timestamp_logging
import warnings
import logging
from openff.evaluator.backends import QueueWorkerResources
from openff.evaluator.backends.dask import DaskLSFBackend
from openff.evaluator.workflow.schemas import ProtocolGroupSchema
from openff.evaluator.server import EvaluatorServer
import copy


class IntegratedOptimizer:
    def __init__(self, input_force_field, test_set_collection, port):
        self.force_field_source = input_force_field
        self.dataset_source = test_set_collection
        self.n_simulations = 0
        self.max_simulations = 0
        self.port = port
        setup_timestamp_logging()
        self.logger = logging.getLogger()

    def prepare_initial_simulations(self, n_samples, smirks, absolute_bounds=None, relative_bounds=None,
                                    include_initial_ff=False):
        if include_initial_ff is True:
            self.logger.info(
                f"Creating a set of {n_samples + 1} force fields for simulation"
            )
        else:
            self.logger.info(
                f"Creating a set of {n_samples} force fields for simulation"
            )
        if relative_bounds is not None:
            if isinstance(relative_bounds, list):
                bounds = relative_bounds
            elif isinstance(relative_bounds, numpy.ndarray):
                bounds = np.load(relative_bounds)
            absolute_ranges = False
        elif absolute_bounds is not None:
            if isinstance(absolute_bounds, list):
                bounds = absolute_bounds
            elif isinstance(absolute_bounds, numpy.ndarray):
                bounds = np.load(absolute_bounds)
        else:
            raise ValueError("Must load either an absolute or relative parameter range")
        os.makedirs(os.path.join('force-fields'))
        self.force_field_directory = os.path.join('force-fields')
        vary_parameters_lhc(self.force_field_source, n_samples, self.force_field_directory, smirks, bounds,
                            nonuniform_ranges=True, absolute_ranges=absolute_ranges)

        for folder in os.listdir(self.force_field_directory):
            shutil.copy2('test-set-collection.json', os.path.join(self.force_field_directory, folder))
        if include_initial_ff is True:
            os.makedirs(os.path.join(self.force_field_directory, str(n_samples + 1)))
            shutil.copy2('test-set-collection.json', os.path.join(self.force_field_directory, str(n_samples + 1)))
            shutil.copy2(self.force_field_source,
                         os.path.join(self.force_field_directory, str(n_samples + 1), 'force-field.offxml'))

    def prepare_single_simulation(self, params, labels):
        params = np.asarray(params)
        df = pandas.DataFrame(params, columns=labels)
        os.makedirs(os.path.join(self.force_field_directory, str(self.n_simulations + 1)))
        forcefield = ForceField(self.force_field_source)
        lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
        for j in range(df.shape[1]):
            smirks = df.columns[j].split('_')[0]
            param = df.columns[j].split('_')[1]
            for lj in lj_params:
                if lj.smirks == smirks:
                    if param == 'epsilon':
                        lj.epsilon = df.values[0, j] * unit.kilocalorie_per_mole
                    elif param == 'rmin':
                        lj.rmin_half = df.values[0, j] * unit.angstrom
        forcefield.to_file(os.path.join(self.force_field_directory, str(self.n_simulations + 1), 'force-field.offxml'))
        shutil.copy2('test-set-collection.json', os.path.join(self.force_field_directory, str(self.n_simulations + 1)))

    def setup_server(self, n_workers=5, cpus_per_worker=1, gpus_per_worker=1, port=8001):

        if n_workers <= 0:
            raise ValueError("The number of workers must be greater than 0")
        if cpus_per_worker <= 0:
            raise ValueError("The number of CPU's per worker must be greater than 0")
        if gpus_per_worker < 0:
            raise ValueError(
                "The number of GPU's per worker must be greater than or equal to 0"
            )
        if 0 < gpus_per_worker != cpus_per_worker:
            raise ValueError(
                "The number of GPU's per worker must match the number of "
                "CPU's per worker."
            )
        if port < 8000:
            raise ValueError("The port number must be greater than or equal to 8000")

            # Set up logging for the evaluator.

        # Set up the directory structure.
        self.working_directory = "../working_directory"

        # Remove any existing data.
        if os.path.isdir(self.working_directory):
            shutil.rmtree(self.working_directory)

        # Set up a backend to run the calculations on with the requested resources.
        if gpus_per_worker <= 0:
            worker_resources = QueueWorkerResources(number_of_threads=cpus_per_worker)
        else:
            worker_resources = QueueWorkerResources(
                number_of_threads=cpus_per_worker,
                number_of_gpus=gpus_per_worker,
                preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                wallclock_time_limit='04:00'
            )
            worker_resources._per_thread_memory_limit *= 5

        # Define the set of commands which will set up the correct environment
        # for each of the workers.
        setup_script_commands = [
            'module load cuda/11.2', 'conda activate LJ_surrogates'
        ]

        # Define extra options to only run on certain node groups
        extra_script_options = [
            '-m "ls-gpu lt-gpu"'
        ]

        self.lsf_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                          maximum_number_of_workers=n_workers,
                                          resources_per_worker=worker_resources,
                                          queue_name='gpuqueue',
                                          setup_script_commands=setup_script_commands,
                                          extra_script_options=extra_script_options)

        # Create an estimation server which will run the calculations.
        self.logger.info(
            f"Starting the server with {n_workers} workers, each with "
            f"{cpus_per_worker} CPUs and {gpus_per_worker} GPUs."
        )

    def submit_requests(self, folder_path, folder_list):
        with EvaluatorServer(calculation_backend=self.lsf_backend,
                             working_directory=self.working_directory,
                             port=self.port,
                             enable_data_caching=False,
                             delete_working_files=True):

            from time import sleep
            from openff.evaluator.client import RequestResult
            requests = {}
            os.makedirs('estimated_results', exist_ok=True)
            self.results_directory = 'estimated_results'

            for subdir in os.listdir(self.force_field_directory):
                if subdir in folder_list:
                    if os.path.exists(os.path.join(folder_path, 'test-set-collection.json') and os.path.exists(
                            os.path.join(folder_path, 'force-field.offxml'))):
                        forcefield = ForceField(
                            os.path.join(self.force_field_directory, subdir, 'force-field.offxml'))
                        property_dataset = PhysicalPropertyDataSet.from_json(
                            os.path.join(self.force_field_directory, subdir, 'test-set-collection.json'))
                    else:
                        raise ValueError(
                            'Folder for request must supply test-set-collection.json and force-field.offxml')

                    if self.n_simulations > self.max_simulations:
                        raise ValueError(
                            f"Unable to request more than {self.max_simulations} simulations.  Please increase the maximum number of simulations")

                    requests[subdir] = self.create_request(property_dataset, forcefield)
                    forcefield.to_file(
                        os.path.join('estimated_results', 'force_field_' + str(subdir) + '.offxml'))

            while len(requests) > 0:

                has_finished = set()

                for subdir, request in requests.items():

                    response, error = request.results(synchronous=False)

                    if (
                            isinstance(response, RequestResult) and
                            len(response.queued_properties) > 0
                    ):
                        # Some properties not estimated yet.
                        continue

                    elif isinstance(response, RequestResult):
                        # ALL PROPERTIES WERE SUCCESSFULLY ESTIMATED
                        response.json(
                            os.path.join('estimated_results', 'estimated_data_set_' + str(subdir) + '.json'))

                    else:
                        print(f"{subdir} FAILED - {error} {response}")

                    has_finished.add(subdir)

                for subdir in has_finished:
                    requests.pop(subdir)

                sleep(60)

    def create_request(self, property_dataset, forcefield):

        density_schema = Density.default_simulation_schema(n_molecules=1000)
        h_mix_schema = EnthalpyOfMixing.default_simulation_schema(n_molecules=1000)

        # h_mix_schema.workflow_schema.protocol_schemas[4].protocol_schemas['production_simulation_component_$(component_replicator)'].inputs['.steps_per_iteration'] *= 4
        # h_mix_schema.workflow_schema.protocol_schemas[11].protocol_schemas['production_simulation_mixture'].inputs['.steps_per_iteration'] *= 4
        # density_schema.workflow_schema.protocol_schemas[4].protocol_schemas['production_simulation'].inputs['.steps_per_iteration'] *= 4
        # Create an options object which defines how the data set should be estimated.
        estimation_options = RequestOptions()
        # Specify that we only wish to use molecular simulation to estimate the data set.
        estimation_options.calculation_layers = ["SimulationLayer"]

        # Add our custom schemas, specifying that the should be used by the 'SimulationLayer'
        estimation_options.add_schema("SimulationLayer", "Density", density_schema)
        estimation_options.add_schema("SimulationLayer", "EnthalpyOfMixing", h_mix_schema)

        connection_options = ConnectionOptions(server_address="localhost", server_port=self.port)
        evaluator_client = EvaluatorClient(connection_options)
        request, _ = evaluator_client.request_estimate(
            property_set=property_dataset,
            force_field_source=forcefield,
            options=estimation_options,
        )
        self.n_simulations += 1
        self.logger.info(
            f"Requesting a simulation of {len(property_dataset.properties)} physical properties."
            f"This is simulation #{self.n_simulations}/{self.max_simulations}"
        )

        return request

    def build_physical_property_surrogate(self, constraint):
        from LJ_surrogates.surrogates.collate_data import collate_physical_property_data

        self.logger.info(f'Building a surrogate model from {self.n_simulations} physical property datasets')

        self.dataplex = collate_physical_property_data(self.results_directory, self.smirks, self.force_field_source,
                                                       self.dataset_source, device='cpu', constraint=constraint)

    def optimize(self):
        pass


class TestOptimizer(IntegratedOptimizer):

    def optimize(self, param_range, smirks):
        from LJ_surrogates.sampling.optimize import ConstrainedGaussianObjectiveFunction
        from scipy.optimize import differential_evolution

        self.max_simulations = 10

        self.setup_server(n_workers=10, cpus_per_worker=1, gpus_per_worker=1, port=self.port)
        with self.lsf_backend:

            self.param_range = param_range
            self.smirks = ['[#18:1]']
            n_samples = 5

            self.prepare_initial_simulations(n_samples=n_samples, smirks=self.smirks, relative_bounds=param_range)

            folder_list = [str(i + 1) for i in range(n_samples)]

            self.submit_requests(folder_path=self.force_field_directory, folder_list=folder_list)

            iter = 0
            objectives = []
            params = []
            while self.n_simulations <= self.max_simulations:
                self.build_physical_property_surrogate()

                self.objective = ConstrainedGaussianObjectiveFunction(self.dataplex.multisurrogate,
                                                                      self.dataplex.properties,
                                                                      self.dataplex.initial_parameters, 0.01)
                self.objective.flatten_parameters()
                self.logger.info(
                    f'Optimization Iteration {iter}: optimizing over a surrogate built from {self.n_simulations} datasets')
                bounds = []
                for column in self.dataplex.parameter_values.columns:
                    minbound = min(self.dataplex.parameter_values[column].values)
                    maxbound = max(self.dataplex.parameter_values[column].values)
                    bounds.append((minbound, maxbound))

                result = differential_evolution(self.objective, bounds)
                self.logger.info(
                    f'Optimization Iteration {iter} complete: Objective function value of {result.fun} and parameters of {result.x}')
                objectives.append(result.fun)
                params.append(result.x)
                if self.n_simulations < self.max_simulations:
                    self.prepare_single_simulation(params=[result.x], labels=self.dataplex.parameter_labels)
                    self.submit_requests(folder_path=self.force_field_directory,
                                         folder_list=[str(self.n_simulations + 1)])
                    iter += 1
                else:
                    break
        self.logger.info(
            f'Optimization complete after {iter} iterations: Objective function value of {result.fun} and parameters of {result.x}')


class SurrogateDESearchOptimizer(IntegratedOptimizer):

    def optimize(self, param_range, smirks, max_simulations, initial_samples, n_workers, use_cached_data=False,
                 cached_data_location=None):
        from LJ_surrogates.sampling.optimize import ForceBalanceObjectiveFunction
        from scipy.optimize import differential_evolution

        self.max_simulations = max_simulations
        self.eta_1 = 0.01
        self.eta_2 = 0.5
        self.bounds_increment = 1.1
        self.max_bounds_expansions = 1
        self.setup_server(n_workers=n_workers, cpus_per_worker=1, gpus_per_worker=1, port=self.port)

        if os.path.exists('estimated_results'):
            os.removedirs('estimated_results')
        if os.path.exists('working_directory'):
            os.removedirs('working_directory')
        if os.path.exists('force-fields'):
            os.removedirs('force-fields')
        if os.path.exists('stored_data'):
            os.removedirs('stored_data')


        with self.lsf_backend:
            self.smirks = smirks
            self.param_range = param_range
            if use_cached_data is True:

                shutil.copytree(cached_data_location,'estimated_results')
                self.n_simulations += int(len(os.listdir('estimated_results'))/2)
                self.results_directory = 'estimated_results'
                os.makedirs(os.path.join('force-fields'))
                self.force_field_directory = os.path.join('force-fields')
            else:
                n_samples = initial_samples

                self.prepare_initial_simulations(n_samples=n_samples, smirks=self.smirks, relative_bounds=param_range,
                                                 include_initial_ff=True)

                folder_list = [str(i + 1) for i in range(n_samples + 1)]

                self.submit_requests(folder_path=self.force_field_directory, folder_list=folder_list)

            iter = 0
            objectives = []
            params = []
            surrogate_rebuild_counter = 0
            constraint = None
            while self.n_simulations <= self.max_simulations:
                if iter == 0:
                    self.build_physical_property_surrogate(constraint=constraint)

                self.objective = ForceBalanceObjectiveFunction(self.dataplex.multisurrogate,
                                                               self.dataplex.properties,
                                                               self.dataplex.initial_parameters,
                                                               self.dataplex.property_labels)
                self.objective.flatten_parameters()

                if iter == 0:
                    self.solution = copy.deepcopy(self.objective.flat_parameters)
                    for i in range(self.dataplex.parameter_values.shape[0]):
                        if np.allclose(self.dataplex.parameter_values.values[i], self.solution):
                            self.logger.info(
                                f'Computing simulation objective for parameter set {self.solution}')
                            self.solution_objective = self.objective.simulation_objective(
                                self.dataplex.property_measurements.values[i])
                            simulation_objective = self.solution_objective
                            break
                if constraint is not None:
                    self.build_physical_property_surrogate(constraint=constraint)
                bounds = []
                for column in self.dataplex.parameter_values.columns:
                    minbound = min(self.dataplex.parameter_values[column].values)
                    maxbound = max(self.dataplex.parameter_values[column].values)
                    bounds.append((minbound, maxbound))
                    self.bounds = np.asarray(bounds)
                self.bounds[:, 0] /= (self.bounds_increment ** (1))
                self.bounds[:, 1] *= (self.bounds_increment ** (1))
                self.logger.info(
                    f'Optimization Iteration {iter}: Initial Solution of {self.solution} with simulation objective of {self.solution_objective}')

                self.logger.info(
                    f'Optimization Iteration {iter}: optimizing over a surrogate built from {self.n_simulations} datasets')

                surrogate_result = differential_evolution(self.objective, self.bounds)
                self.logger.info(
                    f'Surrogate proposes solution with surrogate objective function value of {surrogate_result.fun} and parameters of {surrogate_result.x}')
                predicted_reduction = self.solution_objective - surrogate_result.fun
                if surrogate_result.fun >= self.solution_objective:
                    self.logger.info(
                        f'Surrogate proposed solution has objective {surrogate_result.fun}, >= current simulation objective {self.solution_objective}')
                    from gpytorch.constraints import GreaterThan
                    if surrogate_rebuild_counter == 0:
                        surrogate_rebuild_counter += 1
                        iter += 1
                        self.logger.info(
                            f'Surrogate search unable to find improved candidate solution. Rebuilding surrogate with lengthscale constraints')
                        constraint = GreaterThan(1e-10)
                    elif surrogate_rebuild_counter == 1:
                        surrogate_rebuild_counter += 1
                        iter += 1
                        self.logger.info(
                            f'Surrogate search unable to find improved candidate solution. Rebuilding surrogate with stricter lengthscale constraints')
                        constraint = GreaterThan(1e-5)
                    else:
                        iter += 1
                        self.logger.info(
                            f'Surrogate search unable to find improved candidate solution. Terminating Program')
                        np.save('parameter_vectors.npy', np.asarray(params))
                        np.save('objective_values.npy', np.asarray(objectives))
                        raise ValueError("Unable to find improved solution")
                else:
                    self.logger.info(
                        f'Surrogate proposed solution has objective {surrogate_result.fun}, < current simulation objective {self.solution_objective}')
                    if self.n_simulations < self.max_simulations:
                        self.logger.info(
                            f'Simulating set of parameters from surrogate solution')
                        self.prepare_single_simulation(params=[surrogate_result.x],
                                                       labels=self.dataplex.parameter_labels)
                        self.submit_requests(folder_path=self.force_field_directory,
                                             folder_list=[str(self.n_simulations + 1)])
                        surrogate_rebuild_counter = 0
                        constraint = None
                        self.build_physical_property_surrogate(constraint=constraint)
                        for i in range(self.dataplex.parameter_values.shape[0]):
                            if np.allclose(self.dataplex.parameter_values.values[i], surrogate_result.x):
                                simulation_objective = self.objective.simulation_objective(
                                    self.dataplex.property_measurements.values[i])
                                self.logger.info(
                                    f'Computing simulation objective for parameter set {surrogate_result.x}')
                                break
                        self.logger.info(
                            f'Surrogate proposed solution has simulation objective {simulation_objective}')
                        actual_reduction = self.solution_objective - simulation_objective
                        reduction_ratio = actual_reduction / predicted_reduction
                        if reduction_ratio <= 0:
                            self.logger.info(
                                f'Improvement predicted but not achieved.  Rejecting proposed solution.')
                        elif 0 < reduction_ratio <= self.eta_1:
                            self.solution = surrogate_result.x
                            self.solution_objective = simulation_objective
                            self.logger.info(
                                f'Much less improvement achieved than predicted.  Accepting proposed solution with objective {simulation_objective}')
                        elif self.eta_1 < reduction_ratio <= self.eta_2:
                            self.solution = surrogate_result.x
                            self.solution_objective = simulation_objective
                            self.logger.info(
                                f'Satisfactory prediction.  Accepting proposed solution with objective {simulation_objective}')
                        else:
                            self.solution = surrogate_result.x
                            self.solution_objective = simulation_objective
                            self.logger.info(
                                f'Excellent prediction.  Accepting proposed solution with objective {simulation_objective}')
                        params.append(copy.deepcopy(self.solution))
                        objectives.append(copy.deepcopy(self.solution_objective))
                        iter += 1
                    else:
                        break
        self.logger.info(
            f'Optimization complete after {iter} iterations: Objective function value of {self.solution_objective} and parameters of {self.solution}')
        np.save('parameter_vectors.npy', np.asarray(params))
        np.save('objective_values.npy', np.asarray(objectives))
