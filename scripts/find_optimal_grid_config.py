"""
WIP
"""

import argparse

from ortools.sat.python import cp_model


def find_optimal_grid_config(warp_size=32):
    model = cp_model.CpModel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the optimal grid configuration for a given number of threads per block for CUDA kernel."
    )
    parser.add_argument(
        "--warp-size",
        default=32,
        type=int,
        metavar="INT",
        help="Warp size of the target GPU (default: %(default)i).",
    )
    parser.add_argument(
        "--registers-per-thread",
        "-r",
        type=int,
        metavar="INT",
        default=64,
        help="Number of registers per thread (default: %(default)i).",
    )
    parser.add_argument(
        "--number-of-tasks",
        "-t",
        type=int,
        metavar="INT",
        help="Number of tasks to be executed (the number of invocation of the kernel). Required.",
        required=True,
    )
    kwargs = vars(parser.parse_args())
    find_optimal_grid_config(**kwargs)
