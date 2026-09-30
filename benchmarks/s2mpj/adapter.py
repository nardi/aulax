"""Turn an S2MPJ problem into an aulax problem function.

S2MPJ evaluates its problems with NumPy, which JAX cannot trace, so the adapter
calls it through `jax.pure_callback`. The optimistix minimisers differentiate
with `jax.linearize` and transpose the result, which rules out `jax.custom_vjp`,
and a callback cannot compute a transposable tangent itself. The adapter
therefore uses `jax.custom_jvp`. Its rule fetches the gradient and a dense
Jacobian through a second callback, and computes the tangent with `jnp`
operations, which are linear in the tangent and so can be transposed.
"""

import jax
import jax.numpy as jnp
import numpy as np

import aulax as alx

from ..problems import BenchmarkProblem
from .problems import (
    ConstraintMapping,
    S2mpjProblem,
    has_objective,
    known_objective,
    load_problem,
    objective_and_gradient,
    objective_value,
)


def _values_and_derivatives(problem: S2mpjProblem, has_constraints: bool):
    """Return NumPy functions for the values, and the values with derivatives."""
    size = int(problem.n)

    def values(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        column = np.asarray(x, dtype=float).reshape(size, 1)
        objective = np.asarray(objective_value(problem, column))
        if not has_constraints:
            return objective, np.zeros(0)
        constraints = np.asarray(problem.cx(column), dtype=float).ravel()
        return objective, constraints

    def derivatives(
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        column = np.asarray(x, dtype=float).reshape(size, 1)
        value, gradient = objective_and_gradient(problem, column)
        objective = np.asarray(value)
        if not has_constraints:
            return objective, gradient, np.zeros(0), np.zeros((0, size))
        constraints, jacobian = problem.cJx(column)
        constraints = np.asarray(constraints, dtype=float).ravel()
        if hasattr(jacobian, "toarray"):
            jacobian = jacobian.toarray()
        jacobian = np.asarray(jacobian, dtype=float).reshape(-1, size)
        return objective, gradient, constraints, jacobian

    return values, derivatives


def _lagrangian_hessian(problem: S2mpjProblem, mapping: ConstraintMapping):
    """Return the Hessian of the Lagrangian over the mapped residuals.

    The multipliers come in the order of the flattened residuals. The equality
    residuals are `constraints` then `fixed_variables`, and the inequality
    residuals are `constraints_lower`, `constraints_upper`, `lower_bounds` and
    `upper_bounds`, as `jax.tree.leaves` sorts dictionary keys. Only the
    constraint residuals are nonlinear. A residual $c_l - c$ contributes with a
    negative sign, the others with a positive one.
    """
    size = int(problem.n)
    constraint_count = int(getattr(problem, "m", 0))
    equality_count = mapping.equality.size + mapping.fixed.size
    lower_start = equality_count
    upper_start = lower_start + mapping.lower.size

    def lagrangian_hessian(
        x: np.ndarray, objective_factor: float, multipliers: np.ndarray
    ) -> np.ndarray:
        column = np.asarray(x, dtype=float).reshape(size, 1)
        constraint_multipliers = np.zeros(constraint_count)
        constraint_multipliers[mapping.equality] += multipliers[: mapping.equality.size]
        constraint_multipliers[mapping.lower] -= multipliers[lower_start:upper_start]
        constraint_multipliers[mapping.upper] += multipliers[
            upper_start : upper_start + mapping.upper.size
        ]
        if not has_objective(problem):
            objective_factor = 0.0
        # `LgHxy` returns the Hessian of f + y'c. Scaling y by the objective
        # factor gives the Hessian of the weighted Lagrangian in one call.
        if objective_factor != 0.0:
            _, _, hessian = problem.LgHxy(
                column, constraint_multipliers / objective_factor
            )
            hessian = objective_factor * _dense(hessian)
        elif has_objective(problem):
            _, _, full = problem.LgHxy(column, constraint_multipliers)
            _, _, objective_hessian = problem.fgHx(column)
            hessian = _dense(full) - _dense(objective_hessian)
        else:
            _, _, hessian = problem.LgHxy(column, constraint_multipliers)
            hessian = _dense(hessian)
        return hessian.reshape(size, size)

    return lagrangian_hessian


def _dense(matrix) -> np.ndarray:
    """Return a SciPy sparse matrix or an array as a dense float array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def adapted_problem(name: str, *arguments: object) -> BenchmarkProblem:
    """Load the S2MPJ problem `name` and return it as a benchmark problem."""
    problem = load_problem(name, *arguments)
    mapping = ConstraintMapping.from_problem(problem)
    size = int(problem.n)
    constraint_count = int(getattr(problem, "m", 0))
    values, derivatives = _values_and_derivatives(problem, constraint_count > 0)
    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    vector = jax.ShapeDtypeStruct((size,), jnp.float64)
    constraint_vector = jax.ShapeDtypeStruct((constraint_count,), jnp.float64)
    jacobian_matrix = jax.ShapeDtypeStruct((constraint_count, size), jnp.float64)

    @jax.custom_jvp
    def objective_and_constraints(x):
        return jax.pure_callback(
            values, (scalar, constraint_vector), x, vmap_method="sequential"
        )

    @objective_and_constraints.defjvp
    def objective_and_constraints_jvp(primals, tangents):
        (x,) = primals
        (tangent,) = tangents
        objective, gradient, constraints, jacobian = jax.pure_callback(
            derivatives,
            (scalar, vector, constraint_vector, jacobian_matrix),
            x,
            vmap_method="sequential",
        )
        return (objective, constraints), (gradient @ tangent, jacobian @ tangent)

    def s2mpj_problem(y, args, state):
        objective, constraints = objective_and_constraints(y)
        return alx.Evaluation(
            objective=objective,
            equality=mapping.equality_residuals(y, constraints),
            inequality=mapping.inequality_residuals(y, constraints),
        )

    return BenchmarkProblem(
        name=name,
        problem=s2mpj_problem,
        y0=np.asarray(problem.x0, dtype=float).ravel(),
        objective=known_objective(name),
        lagrangian_hessian=_lagrangian_hessian(problem, mapping),
        tags=frozenset({"s2mpj"}),
    )
