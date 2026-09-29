"""Run solver configurations on a problem set and write one result file each.

Example: `python -m benchmarks.run --problems hock_schittkowski --config baseline`.
"""

import argparse
from pathlib import Path

import jax

from .configs import CONFIGURATIONS
from .problems import PROBLEM_SETS
from .records import RESULTS_DIRECTORY, new_result_file, write_result_file


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
        "--output-dir", type=Path, default=RESULTS_DIRECTORY, help="Where to write."
    )
    arguments = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    problems = PROBLEM_SETS[arguments.problems]()
    if arguments.only:
        problems = [problem for problem in problems if problem.name in arguments.only]

    for configuration_name in arguments.config:
        configuration = CONFIGURATIONS[configuration_name]
        records = []
        for problem in problems:
            record = configuration.solve(problem, arguments.repeats)
            print(
                f"{configuration_name:20} {problem.name:24} {record.outcome:16} "
                f"{record.evaluations}"
            )
            records.append(record)
        result_file = new_result_file(configuration_name, arguments.problems, records)
        path = arguments.output_dir / f"{arguments.problems}-{configuration_name}.json"
        write_result_file(result_file, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
