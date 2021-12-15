import warnings
import logging
from openff.evaluator.utils import setup_timestamp_logging
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions
from openff.evaluator.client import EvaluatorClient, ConnectionOptions
from openff.evaluator.backends import QueueWorkerResources
from openff.evaluator.backends.dask import DaskLSFBackend
from openff.evaluator.server import EvaluatorServer
from openff.units import unit
import os
import numpy as np
import shutil
import time


def run_server(n_workers, cpus_per_worker, gpus_per_worker, files_directory, port):
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
        worker_resources._per_thread_memory_limit *=5

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

    with lsf_backend:

        with EvaluatorServer(calculation_backend=lsf_backend,
                             working_directory=working_directory,
                             port=port,
                             enable_data_caching=False,
                             delete_working_files=True):

            requests = []
            forcefields = []

            from time import sleep
            from openff.evaluator.client import RequestResult
            requests_by_directory = {}

            for subdirectory in os.listdir(files_directory):
                forcefield = SmirnoffForceFieldSource.from_path(
                    os.path.join(files_directory, subdirectory, 'force-field.offxml'))
                property_dataset = PhysicalPropertyDataSet.from_json(
                    os.path.join(files_directory, subdirectory, 'test-set-collection.json'))

                requests_by_directory[subdirectory] = estimate_forcefield_properties(property_dataset, forcefield,port)
                forcefield.to_force_field().to_file(
                    os.path.join('estimated_results', 'force_field_' + str(subdirectory) + '.offxml'))

            while len(requests_by_directory) > 0:

                has_finished = set()

                for subdirectory, request in requests_by_directory.items():

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
                            os.path.join('estimated_results', 'estimated_data_set_' + str(subdirectory) + '.json'))

                    else:
                        print(f"{subdirectory} FAILED - {error} {response}")

                    has_finished.add(subdirectory)

                for subdirectory in has_finished:
                    requests_by_directory.pop(subdirectory)

                sleep(60)


def estimate_forcefield_properties(property_dataset, forcefield, port):
    warnings.filterwarnings('ignore')
    logging.getLogger("openforcefield").setLevel(logging.ERROR)

    setup_timestamp_logging()

    data_set = property_dataset

    force_field_source = forcefield

    density_schema = Density.default_simulation_schema(n_molecules=1000)
    h_mix_schema = EnthalpyOfMixing.default_simulation_schema(n_molecules=1000)

    # Create an options object which defines how the data set should be estimated.
    estimation_options = RequestOptions()
    # Specify that we only wish to use molecular simulation to estimate the data set.
    estimation_options.calculation_layers = ["SimulationLayer"]

    # Add our custom schemas, specifying that the should be used by the 'SimulationLayer'
    estimation_options.add_schema("SimulationLayer", "Density", density_schema)
    estimation_options.add_schema("SimulationLayer", "EnthalpyOfMixing", h_mix_schema)

    connection_options = ConnectionOptions(server_address="localhost", server_port=port)
    evaluator_client = EvaluatorClient(connection_options)

    request, _ = evaluator_client.request_estimate(
        property_set=data_set,
        force_field_source=force_field_source,
        options=estimation_options,
    )

    return request
