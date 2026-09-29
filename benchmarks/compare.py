"""Compare result files of several configurations on the same problem set.

Example: `python -m benchmarks.compare benchmarks/results/hock_schittkowski-*.json`.
The first file is the reference for the evaluation ratios.
"""

import argparse
import math
from pathlib import Path

from .records import ResultFile, SolveRecord, read_result_file


PROFILE_FACTORS = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
"""The factors $\\tau$ at which the performance profile is printed."""


def is_success(record: SolveRecord) -> bool:
    """Return whether a record counts as solved for the comparison."""
    return record.outcome in ("solved", "below_reference")


def solved_cost(record: SolveRecord) -> float:
    """Return the evaluations of a solved record, or infinity if it failed."""
    if not is_success(record) or record.evaluations is None:
        return math.inf
    return float(record.evaluations)


def geometric_mean(values: list[float]) -> float:
    """Return the geometric mean of positive `values`, or NaN if there are none."""
    if not values:
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summary_table(result_files: list[ResultFile]) -> list[str]:
    """Return lines with outcome counts and the evaluation ratio to the first file."""
    reference = {record.problem: record for record in result_files[0].records}
    lines = [
        f"{'configuration':24} {'solved':>6} {'other':>6} {'failed':>6} "
        f"{'errors':>6} {'eval ratio':>10} {'time ratio':>10}"
    ]
    for result_file in result_files:
        counts = {outcome: 0 for outcome in ("solved", "other", "failed", "error")}
        evaluation_ratios = []
        time_ratios = []
        for record in result_file.records:
            if record.outcome == "skipped":
                continue
            if is_success(record):
                counts["solved"] += 1
            elif record.outcome == "other_minimum":
                counts["other"] += 1
            elif record.outcome == "error":
                counts["error"] += 1
            else:
                counts["failed"] += 1
            reference_record = reference.get(record.problem)
            if reference_record is None:
                continue
            if is_success(record) and is_success(reference_record):
                evaluation_ratios.append(
                    solved_cost(record) / solved_cost(reference_record)
                )
                if record.seconds and reference_record.seconds:
                    time_ratios.append(record.seconds / reference_record.seconds)
        lines.append(
            f"{result_file.configuration:24} {counts['solved']:6d} "
            f"{counts['other']:6d} {counts['failed']:6d} {counts['error']:6d} "
            f"{geometric_mean(evaluation_ratios):10.2f} "
            f"{geometric_mean(time_ratios):10.2f}"
        )
    return lines


def performance_profile(result_files: list[ResultFile]) -> list[str]:
    """Return lines with a Dolan-Moré performance profile on evaluations.

    For each factor, a row shows the fraction of problems that each configuration
    solved within that factor of the fewest evaluations any configuration needed.
    """
    problems = sorted(
        {record.problem for file in result_files for record in file.records}
    )
    costs = [
        {record.problem: solved_cost(record) for record in file.records}
        for file in result_files
    ]
    best = {
        problem: min(cost.get(problem, math.inf) for cost in costs)
        for problem in problems
    }
    header = f"{'factor':>8} " + " ".join(
        f"{file.configuration[:14]:>14}" for file in result_files
    )
    lines = [header]
    for factor in PROFILE_FACTORS:
        fractions = []
        for cost in costs:
            within = sum(
                1
                for problem in problems
                if math.isfinite(best[problem])
                and cost.get(problem, math.inf) <= factor * best[problem]
            )
            fractions.append(within / len(problems))
        lines.append(
            f"{factor:8.1f} " + " ".join(f"{fraction:14.2f}" for fraction in fractions)
        )
    return lines


def detail_table(result_files: list[ResultFile]) -> list[str]:
    """Return lines with the outcome and evaluations per problem and file."""
    problems = sorted(
        {record.problem for file in result_files for record in file.records}
    )
    by_problem = [
        {record.problem: record for record in file.records} for file in result_files
    ]
    header = f"{'problem':24} " + " ".join(
        f"{file.configuration[:22]:>22}" for file in result_files
    )
    lines = [header]
    for problem in problems:
        cells = []
        for records in by_problem:
            record = records.get(problem)
            if record is None:
                cells.append(f"{'-':>22}")
            else:
                evaluations = "-" if record.evaluations is None else record.evaluations
                cells.append(f"{record.outcome[:14]:>14} {evaluations:>7}")
        lines.append(f"{problem:24} " + " ".join(cells))
    return lines


def main() -> None:
    """Parse the command line and print the comparison tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Result files.")
    parser.add_argument(
        "--details", action="store_true", help="Also print a row per problem."
    )
    arguments = parser.parse_args()
    result_files = [read_result_file(path) for path in arguments.files]
    problem_sets = {file.problem_set for file in result_files}
    if len(problem_sets) > 1:
        parser.error(f"The files cover different problem sets: {problem_sets}.")

    sections = [
        ("Summary", summary_table(result_files)),
        ("Performance profile on evaluations", performance_profile(result_files)),
    ]
    if arguments.details:
        sections.append(("Per problem", detail_table(result_files)))
    for title, lines in sections:
        print(title)
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
