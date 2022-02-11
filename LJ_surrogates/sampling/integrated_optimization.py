import shutil

import numpy
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet
import pandas
import numpy as np
import os
from LJ_surrogates.parameter_modification import vary_parameters_lhc
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions
from openff.evaluator.client import EvaluatorClient, ConnectionOptions
from simtk import openmm, unit


class IntegratedOptimizer:
    def __init__(self, input_force_field, test_set_collection, port):
        self.force_field_source = input_force_field
        self.dataset_source = test_set_collection
        self.n_simulations = 0
        self.port = port

    def prepare_initial_simulations(self, n_samples, smirks, absolute_bounds=None, relative_bounds=None):
        data_set = PhysicalPropertyDataSet.from_json(self.dataset_source)
        data_set.json(self.dataset_source)

        if relative_bounds is not None:
            if isinstance(relative_bounds,list):
                bounds = relative_bounds
            elif isinstance(relative_bounds,numpy.ndarray):
                bounds = np.load(relative_bounds)
            absolute_ranges = False
        elif absolute_bounds is not None:
            if isinstance(absolute_bounds,list):
                bounds = absolute_bounds
            elif isinstance(absolute_bounds,numpy.ndarray):
                bounds = np.load(absolute_bounds)
        else:
            raise ValueError("Must load either an absolute or relative parameter range")
        os.makedirs(os.path.join('force-fields'))
        self.force_field_directory = os.path.join('force-fields')
        vary_parameters_lhc(self.force_field_source, n_samples, self.force_field_directory, smirks, bounds,
                            nonuniform_ranges=True, absolute_ranges=absolute_ranges)

        for folder in os.listdir(self.force_field_directory):
            shutil.copy2('test-set-collection.json', os.path.join(self.force_field_directory, folder))

    def prepare_single_simulation(self, params, labels):
        params = np.asarray(params)
        df = pandas.DataFrame(params, columns=labels)
        self.n_simulations += 1
        os.makedirs(os.path.join(self.force_field_directory, str(self.n_simulations)))
        forcefield = SmirnoffForceFieldSource(self.force_field_source)
        lj_params = forcefield.get_parameter_handler('vdW', allow_cosmetic_attributes=True)
        for j in range(df.shape[1]):
            smirks = df.columns[j].split('_')[0]
            param = df.columns[j].split('_')[1]
            for lj in lj_params:
                if lj.smirks == smirks:
                    if param == 'epsilon':
                        lj.epsilon = df.values[j] * unit.kilocalorie_per_mole
                    elif param == 'rmin':
                        lj.rmin_half = df.values[j] * unit.angstrom
        forcefield.to_file(os.path.join('optimized_ffs', str(self.n_simulations), 'force-field.offxml'))
        shutil.copy2('test-set-collection.json', os.path.join(self.force_field_directory, str(self.n_simulations)))

    def create_server(self, n_workers=10, cpus_per_worker=1, gpus_per_worker=1, port=8001):

        from openff.evaluator.utils import setup_timestamp_logging
        import warnings
        import logging
        from openff.evaluator.backends import QueueWorkerResources
        from openff.evaluator.backends.dask import DaskLSFBackend
        from openff.evaluator.server import EvaluatorServer

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

        setup_timestamp_logging()
        logger = logging.getLogger()

        # Set up the directory structure.
        working_directory = "../working_directory"

        # Remove any existing data.
        if os.path.isdir(working_directory):
            shutil.rmtree(working_directory)

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

        lsf_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                     maximum_number_of_workers=n_workers,
                                     resources_per_worker=worker_resources,
                                     queue_name='gpuqueue',
                                     setup_script_commands=setup_script_commands,
                                     extra_script_options=extra_script_options)

        # Create an estimation server which will run the calculations.
        logger.info(
            f"Starting the server with {n_workers} workers, each with "
            f"{cpus_per_worker} CPUs and {gpus_per_worker} GPUs."
        )

        self.evaluator_server = EvaluatorServer(calculation_backend=lsf_backend)
        self.evaluator_server.start(asynchronous=True)

    def submit_requests(self, folder_path, folder_list):
        from time import sleep
        from openff.evaluator.client import RequestResult
        requests = {}
        os.makedirs('estimated_results', exist_ok=True)

        for subdir in os.listdir(self.force_field_directory):
            if subdir in folder_list:
                if os.path.exists(os.path.join(folder_path, 'test-set-collection.json') and os.path.exists(
                        os.path.join(folder_path, 'force-field.offxml'))):
                    forcefield = SmirnoffForceFieldSource.from_path(
                        os.path.join(self.force_field_directory, subdir, 'force-field.offxml'))
                    property_dataset = PhysicalPropertyDataSet.from_json(
                        os.path.join(self.force_field_directory, subdir, 'test-set-collection.json'))
                else:
                    raise ValueError('Folder for request must supply test-set-collection.json and force-field.offxml')

            requests[subdir] = self.create_request(property_dataset, forcefield, self.port)
            forcefield.to_force_field().to_file(
                os.path.join('estimated_results', 'force_field_' + str(subdir) + '.offxml'))
        self.n_simulations += len(folder_list)
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

        return request

    def optimize(self):
        self.max_simulations = 25

        self.create_server(n_workers=10, cpus_per_worker=1, gpus_per_worker=1, port=8004)

        param_range = [[0.5, 1.5], [0.95, 1.05], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2],
                       [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2]]
        smirks = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
        n_samples = 10

        self.prepare_initial_simulations(n_samples=n_samples, smirks=smirks, relative_bounds=param_range)

        folder_list = [i for i in range(n_samples)]

        self.submit_requests(folder_path=self.force_field_directory, folder_list=folder_list)

        self.evaluator_server.stop()
