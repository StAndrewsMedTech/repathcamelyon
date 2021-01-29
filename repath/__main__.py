#!/usr/bin/python3

from click import group, version_option, command, argument
from multiprocessing import  set_start_method

import repath.experiments.wang as wang
import repath.experiments.lee as lee
import repath.experiments.liu as liu

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
    