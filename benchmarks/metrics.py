"""How a solve is judged, shared by every solver so the results are comparable."""

import jax
import numpy as np
from jaxtyping import PyTree

import aulax as alx


FEASIBILITY_TOLERANCE = 1e-6
"""The largest constraint violation a solution may have to count as solved."""

OBJECTIVE_RELATIVE_TOLERANCE = 1e-6
"""The relative difference from the known objective that still counts as equal."""

OBJECTIVE_ABSOLUTE_TOLERANCE = 1e-8
"""The absolute difference from the known objective that still counts as equal,
for known objectives near zero."""

OUTCOMES = ("solved", "below_reference", "other_minimum", "failed", "error")
"""The possible outcomes of a solve.

- `"solved"`: the solver converged to a feasible point whose objective matches
  the known one, or no known objective exists.
- `"below_reference"`: the objective is lower than the known one, which usually
  means the known value is rounded or the problem has a better minimum.
- `"other_minimum"`: the objective is higher than the known one, at a feasible
  point where the solver converged, so probably a different local minimum.
- `"failed"`: the solver did not converge, or returned an infeasible point.
- `"error"`: the solve raised an exception.
"""


def flat_leaves(tree: PyTree) -> np.ndarray:
    """Return all leaves of `tree` as one flat NumPy array."""
    leaves = [np.ravel(np.asarray(leaf)) for leaf in jax.tree.leaves(tree)]
    return np.concatenate(leaves) if leaves else np.zeros(0)


def constraint_violation(evaluation: alx.Evaluation) -> float:
    """Return the largest violation of any constraint in `evaluation`."""
    equality = np.abs(flat_leaves(evaluation.equality))
    inequality = np.maximum(flat_leaves(evaluation.inequality), 0.0)
    return float(np.max(np.concatenate([equality, inequality]), initial=0.0))


def classify_outcome(
    converged: bool,
    infeasibility: float,
    objective: float,
    known_objective: float | None,
) -> str:
    """Return the outcome of a finished solve, as defined in `OUTCOMES`."""
    if not converged or not np.isfinite(objective):
        return "failed"
    if infeasibility > FEASIBILITY_TOLERANCE:
        return "failed"
    if known_objective is None:
        return "solved"
    tolerance = max(
        OBJECTIVE_ABSOLUTE_TOLERANCE,
        OBJECTIVE_RELATIVE_TOLERANCE * abs(known_objective),
    )
    if abs(objective - known_objective) <= tolerance:
        return "solved"
    if objective < known_objective:
        return "below_reference"
    return "other_minimum"
