import warnings
import logging
from openff.evaluator.utils import setup_timestamp_logging
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions
from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLSFBackend,QueueWorkerResources
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.client import EvaluatorClient
import os

def estimate_forcefield_properties(property_dataset, forcefield, output_directory):
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


    resources = QueueWorkerResources(number_of_threads=1,
                                     number_of_gpus=1,
                                     preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                     wallclock_time_limit='05:00')

    # Define the set of commands which will set up the correct environment
    # for each of the workers.
    setup_script_commands = [
        'module load cuda/9.2',
    ]

    # Define extra options to only run on certain node groups
    extra_script_options = [
        '-m "ls-gpu lt-gpu"'
    ]

    # Create the backend which will adaptively try to spin up between one and
    # ten workers with the requested resources depending on the calculation load.


    from openff.evaluator.backends.dask import DaskLSFBackend

    lsf_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                 maximum_number_of_workers=10,
                                 resources_per_worker=resources,
                                 queue_name='gpuqueue',
                                 setup_script_commands=setup_script_commands,
                                 extra_script_options=extra_script_options)
    lsf_backend.start()


    evaluator_server = EvaluatorServer(calculation_backend=lsf_backend)
    evaluator_server.start(asynchronous=True)
    evaluator_client = EvaluatorClient()

    request, exception = evaluator_client.request_estimate(
        property_set=data_set,
        force_field_source=force_field_source,
        options=estimation_options,
    )

    assert exception is None

    # Wait for the results.
    results, exception = request.results(synchronous=True, polling_interval=30)
    assert exception is None

    results.estimated_properties.json(os.path.join(output_directory,"estimated_data_set.json"), format=True)

