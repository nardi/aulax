"""Detect simple bounds among the inequality residuals of a problem.

A residual is a bound when its gradient has one nonzero entry and does not change
between two points. The detected bounds become an `alx.Bounds` box, and the
reduced problem contains only the other inequality residuals.
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

import aulax as alx


def split_bounds(
    problem: alx.ProblemFunction[PyTree, PyTree, None, None],
    y0: PyTree,
    args: PyTree,
) -> tuple[alx.Bounds, alx.ProblemFunction[PyTree, PyTree, None, None]]:
    """Return the box on `y` and the problem without the bound residuals.

    The box has the structure of `y0`. Its leaves are infinite where a variable has
    no bound on that side.
    """
    y0 = jax.tree.map(jnp.asarray, y0)
    flat_y0, unravel = ravel_pytree(y0)

    def inequality(flat_y):
        """Return the inequality residuals of the problem as one flat array."""
        evaluation = problem(unravel(flat_y), args, None)
        leaves = jax.tree.leaves(evaluation.inequality)
        if not leaves:
            return jnp.zeros(0, flat_y.dtype)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    lower = np.full(flat_y0.size, -np.inf)
    upper = np.full(flat_y0.size, np.inf)
    residual_y0 = np.asarray(inequality(flat_y0))
    if residual_y0.size == 0:
        return _box(unravel, lower, upper), problem
    jacobian_y0 = np.asarray(jax.jacfwd(inequality)(flat_y0))
    # A random shift shows whether a row of the Jacobian changes with `y`.
    shifted_y = flat_y0 + jnp.asarray(
        np.random.default_rng(0).normal(size=flat_y0.size), flat_y0.dtype
    )
    jacobian_shifted = np.asarray(jax.jacfwd(inequality)(shifted_y))
    kept_rows = []
    for row in range(residual_y0.size):
        nonzero = np.flatnonzero(jacobian_y0[row])
        constant_row = np.allclose(
            jacobian_y0[row], jacobian_shifted[row], rtol=0, atol=1e-12
        )
        if len(nonzero) != 1 or not constant_row:
            kept_rows.append(row)
            continue
        index = nonzero[0]
        coefficient = jacobian_y0[row, index]
        # The residual is `coefficient * y[index] + offset <= 0`.
        offset = residual_y0[row] - coefficient * float(flat_y0[index])
        value = -offset / coefficient
        if coefficient > 0:
            upper[index] = min(upper[index], value)
        else:
            lower[index] = max(lower[index], value)
    kept_rows = np.asarray(kept_rows, dtype=int)

    def problem_without_bounds(y, args, state):
        """Return `problem` evaluated with only the kept inequality residuals."""
        evaluation = problem(y, args, state)
        leaves = jax.tree.leaves(evaluation.inequality)
        if not leaves:
            return evaluation
        residuals = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        return dataclasses.replace(
            evaluation, inequality=residuals[kept_rows] if kept_rows.size else None
        )

    return _box(unravel, lower, upper), problem_without_bounds


def _box(
    unravel: Callable[[jax.Array], PyTree], lower: np.ndarray, upper: np.ndarray
) -> alx.Bounds:
    """Return the flat bounds as an `alx.Bounds` with the structure of `y0`."""
    return alx.Bounds(
        lower=unravel(jnp.asarray(lower)),
        upper=unravel(jnp.asarray(upper)),
    )
