"""Detect simple bounds among the inequality residuals of a problem.

A residual is a bound when its gradient has one nonzero entry and does not change
between two points. The detected bounds are removed from the problem, which then
only keeps the other inequality residuals.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def split_bounds(problem, y0, args):
    """Return the box `(lower, upper)` on the flat `y` and the reduced problem."""
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
        return lower, upper, problem
    jacobian_y0 = np.asarray(jax.jacfwd(inequality)(flat_y0))
    shifted_y = flat_y0 + jnp.asarray(
        np.random.default_rng(0).normal(size=flat_y0.size), flat_y0.dtype
    )
    jacobian_shifted = np.asarray(jax.jacfwd(inequality)(shifted_y))
    keep = []
    for row in range(residual_y0.size):
        nonzero = np.flatnonzero(jacobian_y0[row])
        constant_row = np.allclose(
            jacobian_y0[row], jacobian_shifted[row], rtol=0, atol=1e-12
        )
        if len(nonzero) != 1 or not constant_row:
            keep.append(row)
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
    keep = np.asarray(keep, dtype=int)

    def reduced(y, args, state):
        """Return `problem` evaluated with only the kept inequality residuals."""
        evaluation = problem(y, args, state)
        leaves = jax.tree.leaves(evaluation.inequality)
        if not leaves:
            return evaluation
        residuals = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        return dataclasses.replace(
            evaluation, inequality=residuals[keep] if keep.size else None
        )

    return lower, upper, reduced
