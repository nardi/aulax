from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

from ._problem import Constraints, Evaluation, ProblemFunction


class Bounds(eqx.Module):
    r"""A box $l \le y \le u$ on the variables of a problem.

    Each field is `None` or a pytree whose leaves are scalars or arrays that
    broadcast to the leaves of `y`. A `None` field and an infinite entry both mean
    that there is no bound. The bounds are not part of the augmented Lagrangian,
    and the inner minimiser keeps every iterate inside the box.
    """

    lower: PyTree[Float[Array, "..."]] | None = None
    """The lower bound on `y`, or `None` for no lower bounds."""
    upper: PyTree[Float[Array, "..."]] | None = None
    """The upper bound on `y`, or `None` for no upper bounds."""

    def expand(self, y: PyTree[Float[Array, "..."]]) -> "Bounds":
        """Return the bounds as two pytrees that have exactly the shape of `y`.

        Missing bounds become negative or positive infinity, and scalar leaves are
        broadcast to the shape of the matching leaf of `y`. The leaves have the
        dtype of the leaves of `y`.
        """

        def expand_one(bound: PyTree | None, infinity: float) -> PyTree:
            if bound is None:
                return jax.tree.map(lambda leaf: jnp.full_like(leaf, infinity), y)
            return jax.tree.map(
                lambda leaf, value: jnp.broadcast_to(
                    jnp.asarray(value, leaf.dtype), leaf.shape
                ),
                y,
                bound,
            )

        return Bounds(
            lower=expand_one(self.lower, -jnp.inf),
            upper=expand_one(self.upper, jnp.inf),
        )

    def clip(self, y: PyTree[Float[Array, "..."]]) -> PyTree[Float[Array, "..."]]:
        """Return `y` projected onto the box, which is the nearest point inside it."""
        expanded = self.expand(y)
        return jax.tree.map(jnp.clip, y, expanded.lower, expanded.upper)

    def is_ordered(self) -> Bool[Array, ""]:
        """Return whether every lower bound is at most the matching upper bound.

        This is true when one of the fields is `None`, and false if a bound is
        not a number.
        """
        if self.lower is None or self.upper is None:
            return jnp.array(True)
        comparisons = jax.tree.map(
            lambda lower, upper: jnp.all(jnp.asarray(lower) <= jnp.asarray(upper)),
            self.lower,
            self.upper,
        )
        return jnp.all(jnp.stack(jax.tree.leaves(comparisons)))

    def projected_gradient(
        self, y: PyTree[Float[Array, "..."]], gradient: PyTree[Float[Array, "..."]]
    ) -> PyTree[Float[Array, "..."]]:
        """Return the part of `gradient` that points into the box at `y`.

        These bounds must be expanded to the shape of `y`. An element of `y` that
        is at its lower bound keeps its gradient only when the gradient is
        negative, since only then can a step into the box decrease the function.
        The same holds for the upper bound and a positive gradient. All other
        elements keep their gradient. The largest absolute element of the result
        is zero at a point that satisfies the first-order conditions of the
        bounded problem.
        """

        def project(y, gradient, lower, upper):
            gradient = jnp.where(y <= lower, jnp.minimum(gradient, 0.0), gradient)
            return jnp.where(y >= upper, jnp.maximum(gradient, 0.0), gradient)

        assert self.lower is not None and self.upper is not None
        return jax.tree.map(project, y, gradient, self.lower, self.upper)

    def reduced_costs(
        self, y: PyTree[Float[Array, "..."]], gradient: PyTree[Float[Array, "..."]]
    ) -> "Bounds":
        """Return the multipliers of the bounds that are active at `y`.

        These bounds must be expanded to the shape of `y`, and `gradient` is the
        gradient of the Lagrangian without the bounds. The lower multiplier is
        the positive part of the gradient where `y` is at its lower bound. The
        upper multiplier is the positive part of the negated gradient where `y`
        is at its upper bound. Both are zero elsewhere. The gradient of the
        Lagrangian with the bounds is then the gradient minus the lower
        multiplier plus the upper multiplier, which is zero at a solution.
        """
        assert self.lower is not None and self.upper is not None
        return Bounds(
            lower=jax.tree.map(
                lambda y, gradient, lower: jnp.where(
                    y <= lower, jnp.maximum(gradient, 0.0), 0.0
                ),
                y,
                gradient,
                self.lower,
            ),
            upper=jax.tree.map(
                lambda y, gradient, upper: jnp.where(
                    y >= upper, jnp.maximum(-gradient, 0.0), 0.0
                ),
                y,
                gradient,
                self.upper,
            ),
        )


class BoundedProblem(eqx.Module):
    r"""A problem whose finite bounds are added to its inequality constraints.

    The inequality residuals are the pair of the residuals of `problem` and a
    `Bounds` of the rows $l - y$ and $y - u$. A field of `bounds` that is `None`
    adds no rows. An infinite entry has the constant residual $-1$, which is
    never active, so the multiplier of that row is zero.
    """

    problem: ProblemFunction
    """The problem without the bounds."""
    bounds: Bounds
    """The bounds to add, expanded to the shape of `y`.

    A field that is `None` has no rows.
    """

    def __call__(self, y: PyTree[Array], args: Any, state: Any) -> Evaluation:
        """Evaluate `problem` and add the rows of the bounds to its inequalities."""
        evaluation = self.problem(y, args, state)

        def lower_row(y: Array, lower: Array) -> Array:
            return jnp.where(jnp.isfinite(lower), lower - y, -1.0)

        def upper_row(y: Array, upper: Array) -> Array:
            return jnp.where(jnp.isfinite(upper), y - upper, -1.0)

        rows = Bounds(
            lower=None
            if self.bounds.lower is None
            else jax.tree.map(lower_row, y, self.bounds.lower),
            upper=None
            if self.bounds.upper is None
            else jax.tree.map(upper_row, y, self.bounds.upper),
        )
        return Evaluation(
            objective=evaluation.objective,
            equality=evaluation.equality,
            inequality=(evaluation.inequality, rows),
            state=evaluation.state,
            aux=evaluation.aux,
        )

    def add_zero_rows(self, constraints: Constraints, y: PyTree[Array]) -> Constraints:
        """Return `constraints` with a zero for every row of the bounds."""
        return Constraints(
            equality=constraints.equality,
            inequality=(constraints.inequality, self._rows(y, 0.0)),
        )

    def split_rows(
        self, constraints: Constraints, y: PyTree[Array]
    ) -> tuple[Constraints, Bounds]:
        """Separate the rows of the bounds from `constraints`.

        Returns the constraints of `problem` and a `Bounds` with the values of the
        rows, which have the shape of `y`. A field without rows is zero.
        """
        inequality, rows = constraints.inequality
        zeros = jax.tree.map(jnp.zeros_like, y)
        return Constraints(constraints.equality, inequality), Bounds(
            lower=zeros if rows.lower is None else rows.lower,
            upper=zeros if rows.upper is None else rows.upper,
        )

    def _rows(self, y: PyTree[Array], value: float) -> Bounds:
        """Return a `Bounds` of constants with the structure of the rows."""

        def constants() -> PyTree[Array]:
            return jax.tree.map(lambda leaf: jnp.full_like(leaf, value), y)

        return Bounds(
            lower=None if self.bounds.lower is None else constants(),
            upper=None if self.bounds.upper is None else constants(),
        )
