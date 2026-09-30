"""Problem sets for the benchmarks, by the name the scripts use."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .problems import BenchmarkProblem, hock_schittkowski, scaling


class ProblemSet(Protocol):
    """A named collection of problems that loads one problem at a time."""

    def problem_names(self) -> list[str]:
        """Return the names of the problems, in the order they run."""
        ...

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Return the problem called `name`."""
        ...


@dataclass(frozen=True)
class ListedProblemSet:
    """A problem set whose problems are all defined in Python up front."""

    problems: Callable[[], list[BenchmarkProblem]]
    """Returns every problem in the set."""

    def problem_names(self) -> list[str]:
        """Return the names of the problems, in the order they run."""
        return [problem.name for problem in self.problems()]

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Return the problem called `name`."""
        return next(problem for problem in self.problems() if problem.name == name)


@dataclass(frozen=True)
class S2mpjProblemSet:
    """The S2MPJ problems of one subset of the committed selection.

    Loading a problem instantiates it from the S2MPJ cache, which can take a
    while, so each problem is only loaded when it runs.
    """

    subset: str
    """`general`, `bounds` or `irregular`, as in the selection file."""

    def problem_names(self) -> list[str]:
        """Return the names of the selected problems in this subset."""
        from .s2mpj.select import SELECTION_PATH

        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        return [
            problem["name"]
            for problem in selection["selected"]
            if problem["subset"] == self.subset
        ]

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Return the S2MPJ problem `name` through the JAX adapter."""
        from .s2mpj.adapter import adapted_problem

        return adapted_problem(name)


PROBLEM_SETS: dict[str, ProblemSet] = {
    "hock_schittkowski": ListedProblemSet(lambda: hock_schittkowski.PROBLEMS),
    "scaling": ListedProblemSet(lambda: scaling.PROBLEMS),
    "s2mpj": S2mpjProblemSet("general"),
    "s2mpj_bounds": S2mpjProblemSet("bounds"),
    "s2mpj_irregular": S2mpjProblemSet("irregular"),
}
"""The problem sets by name."""
