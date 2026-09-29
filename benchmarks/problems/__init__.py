"""Problem sets for the benchmarks, by name."""

from collections.abc import Callable

from . import hock_schittkowski, scaling
from ._types import BenchmarkProblem as BenchmarkProblem


PROBLEM_SETS: dict[str, Callable[[], list[BenchmarkProblem]]] = {
    "hock_schittkowski": lambda: hock_schittkowski.PROBLEMS,
    "scaling": lambda: scaling.PROBLEMS,
}
"""Functions that return each problem set, by the name the scripts use."""
