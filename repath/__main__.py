#!/usr/bin/python3

from click import group, version_option, command, argument


@group()
@version_option("1.0.0")
def main():
    pass


@command()
@argument("experiment", required=True)
@argument("step")
def run(experiment: str, step: str) -> None:
    """Run an EXPERIMENT with optional STEP."""
    print(f"{experiment}: {step}")


@command()
@argument("experiment")
def show(experiment: str) -> None:
    """List all the experiments or all the steps for an EXPERIMENT."""
    print(f"{experiment}")


main.add_command(run)
main.add_command(show)


if __name__ == "__main__":
    main()
