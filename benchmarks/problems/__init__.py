"""Problem sets for the benchmarks, by name."""

from collections.abc import Callable

from . import hock_schittkowski
from ._types import BenchmarkProblem as BenchmarkProblem


PROBLEM_SETS: dict[str, Callable[[], list[BenchmarkProblem]]] = {
    "hock_schittkowski": lambda: hock_schittkowski.PROBLEMS,
}
"""Functions that return each problem set, by the name the scripts use."""
