# LJ_surrogates

This repository accompanies this manuscript: [Using physical property surrogate models to perform multi-fidelity global optimization of force field parameters](10.26434/chemrxiv-2022-7bmzv-v2)


Drivers used to prepare parameter sets, process data inputs, build surrogates and run optimizations are found in `LJ_surrogates`.
Specific files used to set up experiments in the paper are found in `projects/pure-only/integrated` and `projects/mixture-only`.
Data files containing the output of all simulations (i.e. force fields and physical property calculation results) used in the optimization, as well any relevant benchmarking outputs, can be found in `data`

## Multi-fidelity optimization workflow

An example surrogate-building workflow (the one used in the paper) is provided in `projects/pure-only/intergated/optimize.py`.  This workflow is intended to run on HPC resources (using GPU resources while requesting simulations using OpenFF Evaluator, and using a single CPU resource to perform the optimization and surrogate building.  In order to adapt this code to specific HPC resources, the `IntegratedOptimizer.setup_server` method may need to be edited; see the [OpenFF Evaluator Documentation](https://docs.openforcefield.org/projects/evaluator/en/stable/gettingstarted/server.html).  The `SurrogateDESearchOptimizer` is initialized with the input forcefield and input experimental physical property dataset.  The optimize method drives the multifidelity optimization, and takes the following inputs:
`param_range`: Relative ranges of each parameter to build the initial parameter space box.

`smirks`: List of vdw SMIRKS types to be optimized

`max_simulations`: The maximum number of simulated data sets that the optimizer will request (initial simulations + additional simulations during the optimization process)

`initial_samples`: The number of initial parameter sets to create with LHS sampling and simulate, before the initial surrogate model is built. The initial force field is also included, but does not count in `initial_samples` (So requesting 9 initial samples will leave you with 9 + 1 = 10 parameter datasets to build surrogates with initially) 

`n_workers`: The number of workers for the Evaluator server to request from HPC resources

`use_cached_data`: Whether to use already existing data (from a previous checkpoint) to build surrogates.  Counts against `max_simulations` (i.e., if you supply 10 dataset/force field pairs in the cached data and you set `max_simulation=15`, the optimizer will only call the simulation level 5 more times.

`cached_data_location`: The filepath to the cached data that should be used. Cached data should be in the form of the `directory` passed to `collate_physical_property_data`.


The script in the example will perform the N=10 optimization described in the paper.

## Building Surrogates

The function to build surrogates are containing in the `surrogates/collate_data.py` module.  Specifically, the `collate_physical_property_data` function will create a `ParameterSetDataMultiplex` object which contains the experimental reference data, simulated parameter sets, physical properties simulated from those parameter sets, and surrogates built from that data.  The surrogates, built with `botorch`, expect N OpenFF Evaluator PhysicalPropertyDataSet objects (see [here](https://github.com/openforcefield/openff-evaluator/blob/main/openff/evaluator/datasets/datasets.py)) as an input (each containing M physical properties) as training outputs, and N parameter vectors (taken from initially supplied `.offxml` SMIRNOFF spec force field files).

To use this function, you should supply
`directory`: A directory containing N estimated physical property datasets, named `estimated_data_set_{n}.json` and N corresponding force fields, labeled `force_field_{n}.offxml`

`smirks`: a list of the SMIRKS types to build surrogate models for

`initial_forcefield`: An `.offxml` file used to extract the initial parameters; can be any simulated forcefield in the dataset.

`properties_filepath`: A path to a `.json` file containing the reference physical properties.

`device`: Which device to use to build the surrogates.  Currently, `'cpu'` is recommended.

## Optimizers

Modifications to the optimization algorithm can be made by creating a class that inherits the base `IntegratedOptimizer` class from the `LJ_surrogates/sampling/integrated_optimization.py` module (e.g. `FooOptimizer(IntegratedOptimizer)`) and overwriting the `optimize` method.

## Bayesian inference

An example notebook demonstrating the use of physical property surrogates to do Bayesian inference for a test problem is available in `projects/examples/argon_single`

