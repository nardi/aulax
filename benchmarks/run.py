"""Run solver configurations on a problem set and write one result file each.

Example: `python -m benchmarks.run --problems hock_schittkowski --config baseline`.
With `--workers`, every problem runs in its own process with a time limit,
which the S2MPJ problems need because their callbacks cannot be interrupted.
"""

import argparse
import dataclasses
from pathlib import Path

import jax
import numpy as np
from jaxtyping import PyTree

from .configs import CONFIGURATIONS
from .problem_sets import PROBLEM_SETS
from .problems import BenchmarkProblem
from .processes import ProcessOutcome, run_in_processes
from .records import (
    RESULTS_DIRECTORY,
    SolveRecord,
    new_result_file,
    read_result_file,
    write_result_file,
)


def cast_to_float32(tree: PyTree) -> PyTree:
    """Return `tree` with every floating point leaf cast to `float32`.

    Leaves with other dtypes, and `None`, are returned unchanged.
    """

    def cast_leaf(leaf):
        """Return `leaf` as `float32` if it holds floating point values."""
        if np.issubdtype(np.asarray(leaf).dtype, np.floating):
            return np.asarray(leaf, dtype=np.float32)
        return leaf

    return jax.tree.map(cast_leaf, tree)


def solve_problem(
    problem_set: str,
    problem_name: str,
    configuration_name: str,
    repeats: int,
    float32: bool = False,
) -> SolveRecord:
    """Load one problem and solve it with one configuration.

    With `float32`, JAX runs with 64-bit floats off and the problem data is cast
    to `float32`. This is the function each worker process runs, so it takes names
    rather than objects, which a process started with `spawn` could not receive.
    """
    jax.config.update("jax_enable_x64", not float32)
    problem = PROBLEM_SETS[problem_set].load_problem(problem_name)
    if float32:
        problem = cast_problem_to_float32(problem)
    return CONFIGURATIONS[configuration_name].solve(problem, repeats)


def cast_problem_to_float32(problem: BenchmarkProblem) -> BenchmarkProblem:
    """Return `problem` with its starting point, arguments and solution in `float32`."""
    return dataclasses.replace(
        problem,
        y0=cast_to_float32(problem.y0),
        args=cast_to_float32(problem.args),
        solution=cast_to_float32(problem.solution),
    )


def run_configuration(
    arguments: argparse.Namespace, names: list[str], configuration_name: str
) -> None:
    """Run one configuration on the named problems and save each record at once."""
    precision_suffix = "-f32" if arguments.float32 else ""
    path = (
        arguments.output_dir
        / f"{arguments.problems}-{configuration_name}{precision_suffix}.json"
    )
    records: list[SolveRecord] = []
    if arguments.resume and path.exists():
        records = read_result_file(path).records
    done = {record.problem for record in records}
    remaining = [name for name in names if name not in done]
    print(f"{configuration_name}: {len(done)} done, {len(remaining)} to run.")

    def save_record(record: SolveRecord) -> None:
        """Add `record` to the result file, so an interrupted run can resume."""
        records.append(record)
        print(
            f"{configuration_name:20} {record.problem:24} {record.outcome:16} "
            f"{record.evaluations}",
            flush=True,
        )
        result_file = new_result_file(configuration_name, arguments.problems, records)
        write_result_file(result_file, path)

    argument_tuples = [
        (
            arguments.problems,
            name,
            configuration_name,
            arguments.repeats,
            arguments.float32,
        )
        for name in remaining
    ]
    if arguments.workers > 0:

        def save_outcome(index: int, outcome: ProcessOutcome[SolveRecord]) -> None:
            """Save the record of a finished process, or an error record."""
            if outcome.result is not None:
                save_record(outcome.result)
                return
            save_record(
                SolveRecord(
                    problem=remaining[index],
                    variables=0,
                    equality_constraints=0,
                    inequality_constraints=0,
                    outcome="error",
                    message=outcome.error or "unknown error",
                )
            )

        run_in_processes(
            solve_problem,
            argument_tuples,
            timeout=arguments.timeout,
            workers=arguments.workers,
            on_outcome=save_outcome,
        )
    else:
        for argument_tuple in argument_tuples:
            save_record(solve_problem(*argument_tuple))
    print(f"Wrote {path}")


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the records in an existing result file and run the rest.",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Run in float32 with 64-bit floats off. Appends -f32 to the file name.",
    )
    arguments = parser.parse_args()

    jax.config.update("jax_enable_x64", not arguments.float32)
    names = PROBLEM_SETS[arguments.problems].problem_names()
    if arguments.only:
        names = [name for name in names if name in arguments.only]

    for configuration_name in arguments.config:
        run_configuration(arguments, names, configuration_name)


if __name__ == "__main__":
    main()
