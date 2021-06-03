#!/usr/bin/python3

from click import group, version_option, command, argument
from multiprocessing import  set_start_method

import repath.experiments.wang as wang
import repath.experiments.lee as lee
import repath.experiments.liu as liu
import repath.experiments.tissuedet as tissue
import repath.experiments.cervical_algorithm1 as cervical1
import repath.experiments.cervical_algorithm2 as cervical2
import repath.experiments.cervical_algorithm3 as cervical3
import repath.experiments.cervical_algorithm4 as cervical4
import repath.experiments.bloodmuc_lev4_rf as bm_lev4

@group()
@version_option("1.0.0")
def main():
    pass


@command()
@argument("experiment")
@argument("step", required=False)
def run(experiment: str, step: str = None) -> None:
    """Run an EXPERIMENT with optional STEP."""
    print(f"{experiment}: {step}")
    eval(f"{experiment}.{step}()")


@command()
@argument("experiment", required=False)
def show(experiment: str) -> None:
    """List all the experiments or all the steps for an EXPERIMENT."""
    print(f"{experiment}")


main.add_command(run)
main.add_command(show)


if __name__ == "__main__":
    set_start_method('spawn')
    main()
    