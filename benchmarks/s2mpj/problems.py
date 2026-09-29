r"""Load S2MPJ problems and map their constraints to aulax's two kinds.

S2MPJ writes every constraint as a range $c_l \\le c(x) \\le c_u$ and every
variable bound as a range $x_l \\le x \\le x_u$, with infinite bounds for absent
sides. aulax wants equality residuals $h(x) = 0$ and inequality residuals
$g(x) \\le 0$. `ConstraintMapping` converts between the two with NumPy index
arrays that are fixed when the problem is loaded.
"""

import importlib
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fetch import CACHE_DIRECTORY, PROBLEM_DIRECTORY


# S2MPJ problems are instances of classes defined in plain Python files, with no
# type information, so they are typed as `Any` throughout.
S2mpjProblem = Any
"""An instance of an S2MPJ problem class."""


def load_problem(name: str, *arguments: object) -> S2mpjProblem:
    """Import the problem file `name` from the cache and instantiate its class.

    The problem files import `s2mpjlib` by its bare name, so the cache directory
    goes on the import path first.
    """
    for directory in (CACHE_DIRECTORY, PROBLEM_DIRECTORY):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    return getattr(module, name)(*arguments)


def known_objective(name: str) -> float | None:
    """Return the optimal objective from the `LO SOLTN` comment, if the file has one."""
    source = (PROBLEM_DIRECTORY / f"{name}.py").read_text()
    match = re.search(r"#\s*LO SOLTN\s+(\S+)", source)
    if match is None:
        return None
    try:
        return float(match.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _column(array: np.ndarray | None, size: int) -> np.ndarray:
    """Return `array` as a flat float array, or an empty one if it is missing."""
    if array is None or size == 0:
        return np.zeros(size)
    return np.asarray(array, dtype=float).ravel()


@dataclass(frozen=True)
class ConstraintMapping:
    """Index arrays that pick the aulax residuals out of S2MPJ's ranges."""

    equality: np.ndarray
    """Indices of constraints with equal lower and upper bounds."""
    upper: np.ndarray
    """Indices of the other constraints with a finite upper bound."""
    lower: np.ndarray
    """Indices of the other constraints with a finite lower bound."""
    fixed: np.ndarray
    """Indices of variables with equal lower and upper bounds."""
    lower_bounded: np.ndarray
    """Indices of the other variables with a finite lower bound."""
    upper_bounded: np.ndarray
    """Indices of the other variables with a finite upper bound."""
    constraint_lower: np.ndarray
    """The lower bounds of all constraints."""
    constraint_upper: np.ndarray
    """The upper bounds of all constraints."""
    variable_lower: np.ndarray
    """The lower bounds of all variables."""
    variable_upper: np.ndarray
    """The upper bounds of all variables."""

    @classmethod
    def from_problem(cls, problem: S2mpjProblem) -> "ConstraintMapping":
        """Build the mapping from the bounds of an S2MPJ problem."""
        m = int(getattr(problem, "m", 0))
        n = int(problem.n)
        constraint_lower = _column(getattr(problem, "clower", None), m)
        constraint_upper = _column(getattr(problem, "cupper", None), m)
        variable_lower = _column(getattr(problem, "xlower", None), n)
        variable_upper = _column(getattr(problem, "xupper", None), n)
        if getattr(problem, "xlower", None) is None:
            variable_lower = np.full(n, -np.inf)
        if getattr(problem, "xupper", None) is None:
            variable_upper = np.full(n, np.inf)
        equal_constraints = constraint_lower == constraint_upper
        fixed_variables = variable_lower == variable_upper
        return cls(
            equality=np.flatnonzero(equal_constraints),
            upper=np.flatnonzero(~equal_constraints & np.isfinite(constraint_upper)),
            lower=np.flatnonzero(~equal_constraints & np.isfinite(constraint_lower)),
            fixed=np.flatnonzero(fixed_variables),
            lower_bounded=np.flatnonzero(
                ~fixed_variables & np.isfinite(variable_lower)
            ),
            upper_bounded=np.flatnonzero(
                ~fixed_variables & np.isfinite(variable_upper)
            ),
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
        )

    def equality_residuals(self, x, c) -> dict[str, Any]:
        """Return the equality residuals, given the variables and constraint values.

        The arguments can be NumPy or JAX arrays, and the result has the same kind.
        """
        return {
            "constraints": c[self.equality] - self.constraint_lower[self.equality],
            "fixed_variables": x[self.fixed] - self.variable_lower[self.fixed],
        }

    def inequality_residuals(self, x, c) -> dict[str, Any]:
        """Return the inequality residuals, given variables and constraint values."""
        return {
            "constraints_upper": c[self.upper] - self.constraint_upper[self.upper],
            "constraints_lower": self.constraint_lower[self.lower] - c[self.lower],
            "lower_bounds": self.variable_lower[self.lower_bounded]
            - x[self.lower_bounded],
            "upper_bounds": x[self.upper_bounded]
            - self.variable_upper[self.upper_bounded],
        }


def evaluate_numpy(
    problem: S2mpjProblem, mapping: ConstraintMapping, x: np.ndarray
) -> tuple[float, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the objective and the mapped residuals at `x`, using S2MPJ directly."""
    column = np.asarray(x, dtype=float).reshape(-1, 1)
    objective = float(np.asarray(problem.fx(column)).ravel()[0])
    if int(getattr(problem, "m", 0)) > 0:
        c = np.asarray(problem.cx(column), dtype=float).ravel()
    else:
        c = np.zeros(0)
    flat = column.ravel()
    return (
        objective,
        mapping.equality_residuals(flat, c),
        mapping.inequality_residuals(flat, c),
    )
