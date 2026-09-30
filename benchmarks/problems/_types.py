from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Array, Float, PyTree

import aulax as alx


@dataclass(frozen=True)
class BenchmarkProblem:
    """A problem function with its starting point and, if known, its solution."""

    name: str
    """A name that is unique within the problem set."""
    problem: alx.ProblemFunction[PyTree[Float[Array, "..."]], PyTree, None, None]
    """The problem function, called with `args` and a `None` state."""
    y0: PyTree[np.ndarray]
    """The starting point, an array or a pytree of arrays."""
    args: PyTree = None
    """The `args` passed to the problem function."""
    solution: PyTree[np.ndarray] | None = None
    """The known minimizer, shaped like `y0`, if there is one and it is unique."""
    objective: float | None = None
    """The known optimal objective value."""
    lagrangian_hessian: Callable[[np.ndarray, float, np.ndarray], np.ndarray] | None = (
        None
    )
    """The Hessian of the Lagrangian, for problems JAX cannot differentiate twice.

    It is called with the flat variables, the objective factor and one multiplier
    per flat residual, equality residuals first, in the order of `jax.tree.leaves`.
    Solvers that need Hessians use it instead of `jax.hessian` when it is set.
    """
    tags: frozenset[str] = field(default_factory=frozenset)
    """Labels for grouping results, like `"equality"` or `"bounds"`."""
