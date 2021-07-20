#!/usr/bin/env python3
import argparse
from LJ_surrogates.run_simulations import run_server


def main(n_workers, cpus_per_worker, gpus_per_worker):
    run_server(n_workers,cpus_per_worker,gpus_per_worker)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Start an EvaluatorServer with a "
        "specified number of workers, each with "
        "access to the specified compute resources.",
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

    args = parser.parse_args()

    main(args.workers, args.cpus_per_worker, args.gpus_per_worker)


    # Create the backend which will adaptively try to spin up between one and
    # ten workers with the requested resources depending on the calculation load.


