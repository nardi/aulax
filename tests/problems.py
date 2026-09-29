"""The Hock-Schittkowski problems that the unit tests solve.

The definitions live in the benchmark suite, and the tests use a fast subset.
"""

from benchmarks.problems import BenchmarkProblem
from benchmarks.problems.hock_schittkowski import PROBLEMS as ALL_PROBLEMS, hs28, hs35


TEST_PROBLEM_NAMES = ("hs6", "hs7", "hs28", "hs21", "hs35", "hs71")
"""The problems the unit tests solve, chosen to run in a few seconds."""

PROBLEMS = [problem for problem in ALL_PROBLEMS if problem.name in TEST_PROBLEM_NAMES]
"""The test problems, in the order of `TEST_PROBLEM_NAMES`."""

__all__ = ["PROBLEMS", "BenchmarkProblem", "hs28", "hs35"]
