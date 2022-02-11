from LJ_surrogates.parameter_modification import vary_parameters_lhc
import os
import pandas
from openff.evaluator.datasets import PhysicalPropertyDataSet
import shutil
import numpy as np
import argparse


def main(n_samples, output_directory, forcefield, data_filepath, bounds, absolute):
    param_range = [[0.5, 1.5], [0.95, 1.05], [0.9, 1.1], [0.95, 1.05], [0.9, 1.1], [0.95, 1.05], [0.95, 1.05],
                   [0.95, 1.05],
                   [0.95, 1.05], [0.95, 1.05], [0.95, 1.05], [0.95, 1.05]]
    smirks = ['[#1:1]-[#6X4]', '[#6:1]', '[#6X4:1]', '[#8:1]', '[#8X2H0+0:1]', '[#8X2H1+0:1]']
    if data_filepath.endswith('.csv'):
        data_csv = pandas.read_csv(data_filepath)
        data_csv['Id'] = data_csv['Id'].astype('string')
        data_set = PhysicalPropertyDataSet.from_pandas(data_csv)
    elif data_filepath.endswith('.json'):
        data_set = PhysicalPropertyDataSet.from_json(data_filepath)
    else:
        raise TypeError('File should either be a dataset in the format of a csv or json')
    data_set.json('test-set-collection.json')
    bounds = np.load(bounds)
    vary_parameters_lhc(forcefield, n_samples, output_directory, smirks, bounds,
                        nonuniform_ranges=True, absolute_ranges=absolute)

    for folder in os.listdir(output_directory):
        shutil.copy2('test-set-collection.json', os.path.join(output_directory, folder))
        shutil.copy2('submit.sh', os.path.join(output_directory, folder))
        shutil.copy2('basic_run.py', os.path.join(output_directory, folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the runfiles for simulating a set of physical properties at a "
                    "variety of force field parameters, in preparation for building a surrogate model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--samples",
        "-nsamples",
        type=int,
        help="The number of LHS parameter sets to create.",
        required=True,
        default=1,
    )

    parser.add_argument(
        "--output_directory",
        "-o",
        type=str,
        help="The name of a parent directory containing all of the ",
        required=True,
    )

    parser.add_argument(
        "--forcefield",
        "-f",
        type=str,
        help="The name of the force field file to perturb with LHS sampling",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Filepath of the physical property dataset (csv) to estimate",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--smirks",
        "-s",
        type=list,
        help="List of SMIRKS types to vary",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--range",
        "-r",
        type=str,
        help="Parameter Range (decimal)",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--absolute",
        "-a",
        type=bool,
        help="Absolute parameter range",
        required=False,
        default=1,
    )
    args = parser.parse_args()

    main(args.samples, args.output_directory, args.forcefield, args.dataset, args.range, args.absolute)


