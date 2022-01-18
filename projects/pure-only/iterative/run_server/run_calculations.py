#!/usr/bin/env python3
import argparse
from LJ_surrogates.run_simulations import run_server
import os


def main(n_workers, cpus_per_worker, gpus_per_worker,ff_directory, port):
    os.makedirs('estimated_results', exist_ok=True)
    run_server(n_workers, cpus_per_worker, gpus_per_worker, ff_directory, port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start an EvaluatorServer with a "
                    "specified number of workers, each with "
                    "access to the specified compute resources."
                    "Then request estimation of all the desired properties",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--workers",
        "-nwork",
        type=int,
        help="The number of compute workers to spawn.",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--cpus_per_worker",
        "-ncpus",
        type=int,
        help="The number CPUs each worker should have acces to. "
             "The server will consume a total of `nwork * ncpus` CPU's.",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--gpus_per_worker",
        "-ngpus",
        type=int,
        help="The number GPUs each worker should have acces to. "
             "The server will consume a total of `nwork * ngpus` GPU's.",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--ff_directory",
        "-dir",
        type=str,
        help="The directory the modified force fields are in ",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="The port number to accept server requests.  Must be above 8000"
             "WARNING: if multiple servers are activated with the same port number,"
             "It may cause requests to be sent to the wrong server.  If using multiple"
             "servers, make sure the port numbers are different",
        required=False,
        default=1,
    )
    args = parser.parse_args()

    main(args.workers, args.cpus_per_worker, args.gpus_per_worker, args.ff_directory, args.port)

# Create the backend which will adaptively try to spin up between one and
# ten workers with the requested resources depending on the calculation load.
