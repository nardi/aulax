"""Run solver configurations on a problem set and write one result file each.

Example: `python -m benchmarks.run --problems hock_schittkowski --config baseline`.
With `--workers`, every problem runs in its own process with a time limit,
which the S2MPJ problems need because their callbacks cannot be interrupted.
"""

import argparse
from pathlib import Path

import jax

from .configs import CONFIGURATIONS
from .problem_sets import PROBLEM_SETS
from .processes import run_in_processes
from .records import (
    RESULTS_DIRECTORY,
    SolveRecord,
    new_result_file,
    write_result_file,
)


def solve_problem(
    problem_set: str, problem_name: str, configuration_name: str, repeats: int
) -> SolveRecord:
    """Load one problem and solve it with one configuration.

    This is the function each worker process runs, so it takes names rather
    than objects, which a process started with `spawn` could not receive.
    """
    jax.config.update("jax_enable_x64", True)
    problem = PROBLEM_SETS[problem_set].load_problem(problem_name)
    return CONFIGURATIONS[configuration_name].solve(problem, repeats)


def main() -> None:
    """Parse the command line, run the solves and write the result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems", required=True, choices=sorted(PROBLEM_SETS), help="Problem set."
    )
    parser.add_argument(
        "--config",
        required=True,
        action="append",
        choices=sorted(CONFIGURATIONS),
        help="Configuration to run. Repeat to run several.",
    )
    parser.add_argument(
        "--only", action="append", help="Run only this problem. Repeat for several."
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed solves per problem."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Run each problem in its own process, this many at a time.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds per problem before its process is ended, with --workers.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIRECTORY, help="Where to write."
    )
    arguments = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    names = PROBLEM_SETS[arguments.problems].problem_names()
    if arguments.only:
        names = [name for name in names if name in arguments.only]

    for configuration_name in arguments.config:
        argument_tuples = [
            (arguments.problems, name, configuration_name, arguments.repeats)
            for name in names
        ]
        if arguments.workers > 0:
            outcomes = run_in_processes(
                solve_problem,
                argument_tuples,
                timeout=arguments.timeout,
                workers=arguments.workers,
            )
            records = [
                outcome.result
                if outcome.result is not None
                else SolveRecord(
                    problem=name,
                    variables=0,
                    equality_constraints=0,
                    inequality_constraints=0,
                    outcome="error",
                    message=outcome.error or "unknown error",
                )
                for name, outcome in zip(names, outcomes, strict=True)
            ]
        else:
            records = [solve_problem(*arguments) for arguments in argument_tuples]
        for record in records:
            print(
                f"{configuration_name:20} {record.problem:24} {record.outcome:16} "
                f"{record.evaluations}"
            )
        result_file = new_result_file(configuration_name, arguments.problems, records)
        path = arguments.output_dir / f"{arguments.problems}-{configuration_name}.json"
        write_result_file(result_file, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
