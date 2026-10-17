"""Turn an S2MPJ problem into an aulax problem function.

S2MPJ evaluates its problems with NumPy, which JAX cannot trace, so the adapter
calls it through `jax.pure_callback`. The optimistix minimisers differentiate
with `jax.linearize` and transpose the result, which rules out `jax.custom_vjp`,
and a callback cannot compute a transposable tangent itself. The adapter
therefore uses `jax.custom_jvp`. Its rule fetches the gradient and a dense
Jacobian through a second callback, and computes the tangent with `jnp`
operations, which are linear in the tangent and so can be transposed.

A minimiser that uses exact Hessians differentiates the gradient and the
Jacobian once more. They come from a second `jax.custom_jvp` function, whose
rule fetches the dense Hessian of the objective and of every constraint.
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


def _second_derivatives(problem: S2mpjProblem, has_constraints: bool):
    """Return a NumPy function for the Hessians of the objective and constraints."""
    size = int(problem.n)
    constraint_count = int(getattr(problem, "m", 0)) if has_constraints else 0

    def second_derivatives(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        column = np.asarray(x, dtype=float).reshape(size, 1)
        if has_objective(problem):
            objective_hessian = _dense(problem.fgHx(column)[2]).reshape(size, size)
        else:
            objective_hessian = np.zeros((size, size))
        constraint_hessians = np.zeros((constraint_count, size, size))
        if has_constraints:
            for index, hessian in enumerate(problem.cJHx(column)[2]):
                constraint_hessians[index] = _dense(hessian).reshape(size, size)
        return objective_hessian, constraint_hessians

    return second_derivatives


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
    second_derivatives = _second_derivatives(problem, constraint_count > 0)
    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    vector = jax.ShapeDtypeStruct((size,), jnp.float64)
    constraint_vector = jax.ShapeDtypeStruct((constraint_count,), jnp.float64)
    jacobian_matrix = jax.ShapeDtypeStruct((constraint_count, size), jnp.float64)
    hessian_matrix = jax.ShapeDtypeStruct((size, size), jnp.float64)
    constraint_hessians = jax.ShapeDtypeStruct(
        (constraint_count, size, size), jnp.float64
    )

    @jax.custom_jvp
    def objective_and_constraints(x):
        return jax.pure_callback(
            values, (scalar, constraint_vector), x, vmap_method="sequential"
        )

    @jax.custom_jvp
    def first_derivatives(x):
        return jax.pure_callback(
            derivatives,
            (scalar, vector, constraint_vector, jacobian_matrix),
            x,
            vmap_method="sequential",
        )

    @first_derivatives.defjvp
    def first_derivatives_jvp(primals, tangents):
        (x,) = primals
        (tangent,) = tangents
        objective_hessian, constraint_hessian_tensor = jax.pure_callback(
            second_derivatives,
            (hessian_matrix, constraint_hessians),
            x,
            vmap_method="sequential",
        )
        primal_output = first_derivatives(x)
        _, gradient, _, jacobian = primal_output
        return primal_output, (
            gradient @ tangent,
            objective_hessian @ tangent,
            jacobian @ tangent,
            jnp.einsum("mij,j->mi", constraint_hessian_tensor, tangent),
        )

    @objective_and_constraints.defjvp
    def objective_and_constraints_jvp(primals, tangents):
        (x,) = primals
        (tangent,) = tangents
        objective, gradient, constraints, jacobian = first_derivatives(x)
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
