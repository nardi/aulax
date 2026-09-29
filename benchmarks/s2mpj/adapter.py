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
from .problems import ConstraintMapping, S2mpjProblem, known_objective, load_problem


def _values_and_derivatives(problem: S2mpjProblem, has_constraints: bool):
    """Return NumPy functions for the values, and the values with derivatives."""
    size = int(problem.n)

    def values(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        column = np.asarray(x, dtype=float).reshape(size, 1)
        objective = np.asarray(problem.fx(column), dtype=float).reshape(())
        if not has_constraints:
            return objective, np.zeros(0)
        constraints = np.asarray(problem.cx(column), dtype=float).ravel()
        return objective, constraints

    def derivatives(
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        column = np.asarray(x, dtype=float).reshape(size, 1)
        objective, gradient = problem.fgx(column)
        objective = np.asarray(objective, dtype=float).reshape(())
        gradient = np.asarray(gradient, dtype=float).ravel()
        if not has_constraints:
            return objective, gradient, np.zeros(0), np.zeros((0, size))
        constraints, jacobian = problem.cJx(column)
        constraints = np.asarray(constraints, dtype=float).ravel()
        if hasattr(jacobian, "toarray"):
            jacobian = jacobian.toarray()
        jacobian = np.asarray(jacobian, dtype=float).reshape(-1, size)
        return objective, gradient, constraints, jacobian

    return values, derivatives


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
        tags=frozenset({"s2mpj"}),
    )
