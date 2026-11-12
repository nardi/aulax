"""Summarise result files against a reference result file.

Example: `python -m benchmarks.summary benchmarks/results/hock_schittkowski-*.json`.
The reference is the first file unless `--reference` names another file of the
list. The evaluation ratios divide by the reference, over the problems that both
files solve.
"""

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from .compare import geometric_mean, is_success, solved_cost
from .records import ResultFile, read_result_file


@dataclass(frozen=True)
class ConfigurationSummary:
    """The solved count and evaluation ratios of one result file."""

    label: str
    """The file name without the problem set, so `-f32` results have their own row."""
    solved_count: int
    """The number of records that count as solved."""
    record_count: int
    """The number of records in the file."""
    median_ratio: float
    """The median of the evaluations relative to the reference, or NaN if none."""
    geometric_mean_ratio: float
    """The geometric mean of the evaluations relative to the reference, or NaN."""
    missed_problems: list[str]
    """The problems that the reference solves and this file does not, sorted."""


def label_for(path: Path, result_file: ResultFile) -> str:
    """Return the name a result file is listed under."""
    return path.stem.removeprefix(f"{result_file.problem_set}-")


def summarise(
    reference: ResultFile, result_file: ResultFile, label: str
) -> ConfigurationSummary:
    """Return the summary of `result_file` against `reference`.

    Ratios use the problems that both files solve, and a ratio is skipped when
    either evaluation count is missing.
    """
    reference_records = {record.problem: record for record in reference.records}
    records = {record.problem: record for record in result_file.records}
    reference_solved = {
        name for name, record in reference_records.items() if is_success(record)
    }
    solved = {name for name, record in records.items() if is_success(record)}

    ratios = []
    for name in sorted(solved & reference_solved):
        ratio = solved_cost(records[name]) / solved_cost(reference_records[name])
        if math.isfinite(ratio):
            ratios.append(ratio)
    return ConfigurationSummary(
        label=label,
        solved_count=len(solved),
        record_count=len(records),
        median_ratio=statistics.median(ratios) if ratios else math.nan,
        geometric_mean_ratio=geometric_mean(ratios),
        missed_problems=sorted(reference_solved - solved),
    )


def summary_lines(summaries: list[ConfigurationSummary]) -> list[str]:
    """Return the table of counts and ratios, then the missed problems per row."""
    lines = [
        f"{'configuration':24} {'solved':>10} {'median ratio':>13} "
        f"{'geometric mean':>15}"
    ]
    for summary in summaries:
        solved = f"{summary.solved_count}/{summary.record_count}"
        lines.append(
            f"{summary.label:24} {solved:>10} {summary.median_ratio:13.2f} "
            f"{summary.geometric_mean_ratio:15.2f}"
        )
    lines.append("")
    lines.append("Solved by the reference but not by the configuration:")
    for summary in summaries:
        missed = ", ".join(summary.missed_problems) or "none"
        lines.append(f"{summary.label}: {missed}")
    return lines


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Result files.")
    parser.add_argument(
        "--reference",
        type=Path,
        help="The result file to compare with. Defaults to the first file.",
    )
    arguments = parser.parse_args(argv)

    loaded = [(path, read_result_file(path)) for path in arguments.files]
    problem_sets = {result_file.problem_set for _, result_file in loaded}
    if len(problem_sets) > 1:
        parser.error(f"The files cover different problem sets: {problem_sets}.")
    reference_path = arguments.reference or arguments.files[0]
    if reference_path not in arguments.files:
        parser.error("The reference must be one of the result files.")
    reference = dict(loaded)[reference_path]

    summaries = [
        summarise(reference, result_file, label_for(path, result_file))
        for path, result_file in loaded
    ]
    print("\n".join(summary_lines(summaries)))


if __name__ == "__main__":
    main()
