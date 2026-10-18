import functools
import math
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree, Scalar

from ._misc import tree_max_norm
from ._problem import Args, Constraints, Evaluation, ProblemFunction, State, Y


class Scaling(eqx.Module):
    r"""Positive factors that rescale the objective and every constraint element.

    The solver minimizes $s_f f$ subject to $s_h h = 0$ and $s_g g \le 0$, which
    has the same solution as the original problem. The multipliers of the scaled
    problem relate to the original ones by $\lambda = s_h \lambda' / s_f$.
    """

    objective: Scalar
    """The factor $s_f$ for the objective."""
    constraints: Constraints
    """The factors $s_h$ and $s_g$, one for each constraint element."""

    def scale_evaluation(self, evaluation: Evaluation) -> Evaluation:
        """Return `evaluation` with the objective and residuals scaled."""
        return Evaluation(
            objective=evaluation.objective * self.objective,
            equality=jax.tree.map(
                jnp.multiply, evaluation.equality, self.constraints.equality
            ),
            inequality=jax.tree.map(
                jnp.multiply, evaluation.inequality, self.constraints.inequality
            ),
            state=evaluation.state,
            aux=evaluation.aux,
        )

    def unscale_evaluation(self, evaluation: Evaluation) -> Evaluation:
        """Return the original evaluation of a scaled `evaluation`."""
        return Evaluation(
            objective=evaluation.objective / self.objective,
            equality=jax.tree.map(
                jnp.divide, evaluation.equality, self.constraints.equality
            ),
            inequality=jax.tree.map(
                jnp.divide, evaluation.inequality, self.constraints.inequality
            ),
            state=evaluation.state,
            aux=evaluation.aux,
        )

    def scale_multipliers(self, multipliers: Constraints) -> Constraints:
        """Return the multipliers of the scaled problem for original `multipliers`."""
        return self._map_multipliers(multipliers, lambda m, s: m * self.objective / s)

    def unscale_multipliers(self, multipliers: Constraints) -> Constraints:
        """Return the original multipliers for those of the scaled problem."""
        return self._map_multipliers(multipliers, lambda m, s: m * s / self.objective)

    def _map_multipliers(self, multipliers: Constraints, function) -> Constraints:
        return Constraints(
            equality=jax.tree.map(
                function, multipliers.equality, self.constraints.equality
            ),
            inequality=jax.tree.map(
                function, multipliers.inequality, self.constraints.inequality
            ),
        )


class ProblemScaling(Protocol):
    """A rule that chooses the scaling of a problem from its start point."""

    def scale_factors(
        self,
        problem: ProblemFunction[Y, Args, State, object],
        y0: Y,
        args: Args,
        state: State,
        evaluation0: Evaluation,
        objective_gradient: PyTree[Float[Array, "..."]],
    ) -> Scaling:
        """Return the scaling factors.

        `evaluation0` and `objective_gradient` are the evaluation of `problem`
        at `y0` and the gradient of its objective there.
        """
        ...


def _row_maxima(jacobian: PyTree, residuals: PyTree) -> PyTree:
    """Return the largest absolute entry in every row of a Jacobian.

    `jacobian` has the structure of `residuals` at its top, and below each
    residual leaf a pytree shaped like `y`. The result is shaped like `residuals`.
    """

    def row_maxima(residual: Array, columns: PyTree) -> Array:
        residual_shape = jnp.shape(residual)
        # The number of columns is spelled out because `-1` cannot be inferred
        # for a residual group with no elements.
        maxima = [
            jnp.max(
                jnp.abs(block).reshape(
                    (*residual_shape, math.prod(block.shape[len(residual_shape) :]))
                ),
                axis=-1,
                initial=0.0,
            )
            for block in jax.tree.leaves(columns)
        ]
        return functools.reduce(jnp.maximum, maxima, jnp.zeros_like(residual))

    return jax.tree.map(row_maxima, residuals, jacobian)


class GradientScaling(eqx.Module):
    r"""Scales the problem so that no gradient at the start point is large.

    The objective and every constraint element get the factor
    $\min(1, \kappa / \lVert \nabla c \rVert_\infty)$ with the gradient taken at
    `y0`. This is the default scaling of IPOPT. A badly scaled constraint makes
    the penalty term ill-conditioned, and the scaling spares the solver the penalty
    increases that would otherwise compensate.

    The Jacobian of the constraints at `y0` is computed in full when it has at most
    `dense_entries` entries, which costs as many derivative passes as the smaller
    of the number of variables and the number of constraint elements. Larger
    problems estimate the norm of every row from a few Jacobian-vector products
    with random vectors instead, since $\mathbb{E}[(J z)_i^2] = \lVert J_i \rVert_2^2$
    for a standard normal $z$. The estimate is a rough scale, which is all the
    factor needs.
    """

    limit: float = eqx.field(default=100.0, static=True)
    r"""The largest gradient entry $\kappa$ that the scaling leaves alone."""
    dense_entries: int = eqx.field(default=4_000_000, static=True)
    """The most entries that the Jacobian may have to be computed in full."""
    probes: int = eqx.field(default=32, static=True)
    """The number of random vectors for the estimate of the row norms."""

    def _probed_row_norms(self, constraint_residuals: Callable, y0: PyTree) -> PyTree:
        """Estimate the 2-norm of every row of the Jacobian with random probes."""
        flat, unravel = ravel_pytree(y0)
        keys = jax.random.split(jax.random.key(0), self.probes)

        def product(key: Array) -> PyTree:
            tangent = unravel(jax.random.normal(key, flat.shape, flat.dtype))
            return jax.jvp(constraint_residuals, (y0,), (tangent,))[1]

        products = jax.vmap(product)(keys)
        return jax.tree.map(lambda leaf: jnp.sqrt(jnp.mean(leaf**2, axis=0)), products)

    def scale_factors(
        self,
        problem: ProblemFunction[Y, Args, State, object],
        y0: Y,
        args: Args,
        state: State,
        evaluation0: Evaluation,
        objective_gradient: PyTree[Float[Array, "..."]],
    ) -> Scaling:
        """Return factors from the gradients of the objective and constraints."""
        dtype = jnp.result_type(evaluation0.objective)

        def factor(largest: Array) -> Array:
            largest = jnp.asarray(largest, dtype)
            return jnp.where(largest > self.limit, self.limit / largest, 1.0)

        def constraint_residuals(y: Y) -> tuple[PyTree, PyTree]:
            evaluation = problem(y, args, state)
            return evaluation.equality, evaluation.inequality

        residuals = (evaluation0.equality, evaluation0.inequality)
        residual_count = sum(jnp.size(leaf) for leaf in jax.tree.leaves(residuals))
        if residual_count == 0:
            factors = jax.tree.map(lambda leaf: jnp.ones_like(leaf, dtype), residuals)
        else:
            variable_count = sum(jnp.size(leaf) for leaf in jax.tree.leaves(y0))
            if variable_count * residual_count <= self.dense_entries:
                differentiate = (
                    jax.jacfwd if variable_count <= residual_count else jax.jacrev
                )
                jacobian = differentiate(constraint_residuals)(y0)
                norms = _row_maxima(jacobian, residuals)
            else:
                norms = self._probed_row_norms(constraint_residuals, y0)
            factors = jax.tree.map(factor, norms)
        return Scaling(
            objective=factor(tree_max_norm(objective_gradient)),
            constraints=Constraints(equality=factors[0], inequality=factors[1]),
        )
