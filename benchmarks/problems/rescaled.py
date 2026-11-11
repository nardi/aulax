"""Benchmark problems with random scale factors on their variables and outputs."""

import dataclasses
import zlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ._types import BenchmarkProblem


def rescaled(problem: BenchmarkProblem, spread: float = 2.0) -> BenchmarkProblem:
    """Return `problem` in the variables `z = y / variable_factor`, with scaled outputs.

    Every factor is `10**u` with `u` uniform in `[-spread, spread]`. The generator
    is seeded with the name of `problem`, so the factors are the same on every run.
    """
    generator = np.random.default_rng(zlib.crc32(problem.name.encode()))
    y0 = jax.tree.map(jnp.asarray, problem.y0)
    flat_y0, unravel = ravel_pytree(y0)
    variable_factor = 10.0 ** generator.uniform(-spread, spread, flat_y0.size)
    objective_factor = float(10.0 ** generator.uniform(-spread, spread))
    evaluation_y0 = problem.problem(y0, problem.args, None)
    equality_size = sum(
        np.size(leaf) for leaf in jax.tree.leaves(evaluation_y0.equality)
    )
    inequality_size = sum(
        np.size(leaf) for leaf in jax.tree.leaves(evaluation_y0.inequality)
    )
    equality_factor = 10.0 ** generator.uniform(-spread, spread, equality_size)
    inequality_factor = 10.0 ** generator.uniform(-spread, spread, inequality_size)

    def flatten_leaves(tree):
        """Return the leaves of `tree` concatenated into one flat array, or `None`."""
        leaves = jax.tree.leaves(tree)
        if not leaves:
            return None
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    def scaled_problem(scaled_point, args, state):
        """Evaluate `problem` at the unscaled point and scale its outputs."""
        flat_scaled_point, _ = ravel_pytree(scaled_point)
        evaluation = problem.problem(
            unravel(flat_scaled_point * variable_factor), args, state
        )
        equality = flatten_leaves(evaluation.equality)
        inequality = flatten_leaves(evaluation.inequality)
        return dataclasses.replace(
            evaluation,
            objective=objective_factor * evaluation.objective,
            equality=None if equality is None else equality * equality_factor,
            inequality=None if inequality is None else inequality * inequality_factor,
        )

    solution = None
    if problem.solution is not None:
        solution_flat, _ = ravel_pytree(jax.tree.map(jnp.asarray, problem.solution))
        solution = np.asarray(solution_flat) / variable_factor
    return dataclasses.replace(
        problem,
        problem=scaled_problem,
        y0=np.asarray(flat_y0) / variable_factor,
        solution=solution,
        objective=None
        if problem.objective is None
        else objective_factor * problem.objective,
        # The Hessian of the original problem does not apply to the scaled variables.
        lagrangian_hessian=None,
    )
