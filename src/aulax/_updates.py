from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Scalar

from ._misc import dtype_default, tree_any, tree_full_like, tree_max_norm, tree_sum
from ._problem import Constraints, Evaluation


def _constraint_dtype(evaluation: Evaluation) -> DTypeLike:
    """Return the dtype the solver uses for multipliers and penalties."""
    return jnp.result_type(evaluation.objective)


class MultiplierUpdate(Protocol):
    """A rule that updates the Lagrange multipliers after each inner solve."""

    def safeguard_multipliers(
        self, multipliers: Constraints, evaluation: Evaluation
    ) -> Constraints:
        """Return the multipliers to use in the next augmented Lagrangian."""
        ...

    def update_multipliers(
        self, multipliers: Constraints, penalty: Constraints, evaluation: Evaluation
    ) -> Constraints:
        """Return new multiplier estimates from an inner solution.

        `multipliers` are the ones used in the augmented Lagrangian that was just
        minimized, and `evaluation` is the problem at its minimizer.
        """
        ...


class SafeguardedMultiplierUpdate(eqx.Module):
    r"""The first-order multiplier update with a safeguarding box.

    The estimates are $\lambda + \rho h$ and $\max(0, \mu + \rho g)$. Before they
    go into the next augmented Lagrangian, they are clipped to
    $[-\bar\lambda, \bar\lambda]$ and $[0, \bar\mu]$. The global convergence
    theory of the method depends on this box, and it keeps the multipliers
    finite in float32.
    """

    equality_bound: float | None = None
    r"""The bound $\bar\lambda$ on the equality multipliers.

    `None` selects $10^{20}$ in float64 and $10^{6}$ in float32.
    """
    inequality_bound: float | None = None
    r"""The bound $\bar\mu$ on the inequality multipliers.

    `None` selects $10^{20}$ in float64 and $10^{6}$ in float32.
    """

    def safeguard_multipliers(
        self, multipliers: Constraints, evaluation: Evaluation
    ) -> Constraints:
        """Clip the multipliers to the safeguarding box."""
        dtype = _constraint_dtype(evaluation)
        equality_bound = (
            dtype_default(dtype, 1e20, 1e6)
            if self.equality_bound is None
            else self.equality_bound
        )
        inequality_bound = (
            dtype_default(dtype, 1e20, 1e6)
            if self.inequality_bound is None
            else self.inequality_bound
        )
        return Constraints(
            equality=jax.tree.map(
                lambda multiplier: jnp.clip(
                    multiplier, -equality_bound, equality_bound
                ),
                multipliers.equality,
            ),
            inequality=jax.tree.map(
                lambda multiplier: jnp.clip(multiplier, 0.0, inequality_bound),
                multipliers.inequality,
            ),
        )

    def update_multipliers(
        self, multipliers: Constraints, penalty: Constraints, evaluation: Evaluation
    ) -> Constraints:
        """Return the first-order multiplier estimates."""
        return Constraints(
            equality=jax.tree.map(
                lambda multiplier, rho, residual: multiplier + rho * residual,
                multipliers.equality,
                penalty.equality,
                evaluation.equality,
            ),
            inequality=jax.tree.map(
                lambda multiplier, rho, residual: jnp.maximum(
                    0.0, multiplier + rho * residual
                ),
                multipliers.inequality,
                penalty.inequality,
                evaluation.inequality,
            ),
        )


class PenaltyUpdate(Protocol):
    """A rule that sets the initial penalty and grows it between inner solves."""

    def initial_penalty(self, evaluation: Evaluation) -> Constraints:
        """Return the penalty for the first inner solve, one value per constraint."""
        ...

    def update_penalty(
        self,
        penalty: Constraints,
        infeasibility: Constraints,
        previous_infeasibility: Constraints,
    ) -> tuple[Constraints, Bool[Array, ""]]:
        """Return the penalty for the next inner solve.

        The infeasibility arguments are the elementwise measures from
        `penalty_infeasibility` after the current and the previous inner solve.
        The second output tells whether the penalty had to grow but has reached
        its maximum, which the solver takes as a sign of an infeasible problem.
        """
        ...


class AlgencanPenaltyUpdate(eqx.Module):
    r"""The penalty rule of ALGENCAN.

    The penalty grows by `growth_factor` when the infeasibility has not decreased
    to at most `decrease_factor` times its previous value. By default one penalty
    covers all constraints and the infeasibility is the largest element. With
    `per_constraint=True`, every constraint element has its own penalty, which
    grows based on that element's infeasibility alone.

    The initial penalty follows Birgin and Martínez (2014), and balances the
    objective against the initial constraint violation:

    $$
    \rho_0 = \max\left(10^{-8}, \min\left(
        \frac{10 \max(1, \lvert f \rvert)}
             {\max(1, \frac{1}{2} \lVert h \rVert^2
                      + \frac{1}{2} \lVert \max(0, g) \rVert^2)},
        10^{8} \right)\right).
    $$
    """

    growth_factor: float = 10.0
    r"""The factor $\gamma$ by which the penalty grows."""
    decrease_factor: float = 0.5
    r"""The factor $\tau$ by which the infeasibility must decrease between inner
    solves to keep the penalty the same."""
    per_constraint: bool = eqx.field(default=False, static=True)
    """Whether every constraint element has its own penalty."""
    initial_value: float | None = None
    """The initial penalty. `None` selects the ALGENCAN formula."""
    maximum_value: float | None = None
    r"""The largest penalty. `None` selects $10^{12}$ in float64 and $10^{6}$ in
    float32."""

    def initial_penalty(self, evaluation: Evaluation) -> Constraints:
        """Return the initial penalty for every constraint element."""
        if self.initial_value is None:
            squared_violation = 0.5 * tree_sum(
                jnp.square, evaluation.equality
            ) + 0.5 * tree_sum(
                lambda residual: jnp.maximum(0.0, residual) ** 2,
                evaluation.inequality,
            )
            value = 10.0 * jnp.maximum(1.0, jnp.abs(evaluation.objective))
            value = value / jnp.maximum(1.0, squared_violation)
            value = jnp.clip(value, 1e-8, 1e8)
        else:
            value = jnp.asarray(self.initial_value)
        value = value.astype(_constraint_dtype(evaluation))
        return Constraints(
            equality=tree_full_like(evaluation.equality, value),
            inequality=tree_full_like(evaluation.inequality, value),
        )

    def update_penalty(
        self,
        penalty: Constraints,
        infeasibility: Constraints,
        previous_infeasibility: Constraints,
    ) -> tuple[Constraints, Bool[Array, ""]]:
        """Grow the penalty where the infeasibility did not decrease enough."""
        dtype = jnp.result_type(*jax.tree.leaves(penalty))
        maximum = (
            dtype_default(dtype, 1e12, 1e6)
            if self.maximum_value is None
            else self.maximum_value
        )
        if self.per_constraint:
            grow = jax.tree.map(
                lambda current, previous: current > self.decrease_factor * previous,
                infeasibility,
                previous_infeasibility,
            )
        else:
            grow_all = tree_max_norm(
                infeasibility
            ) > self.decrease_factor * tree_max_norm(previous_infeasibility)
            grow = jax.tree.map(lambda _: grow_all, infeasibility)
        new_penalty = jax.tree.map(
            lambda rho, grow_element: jnp.where(
                grow_element, jnp.minimum(self.growth_factor * rho, maximum), rho
            ),
            penalty,
            grow,
        )
        stalled = jax.tree.map(
            lambda rho, grow_element: grow_element & (rho >= maximum),
            penalty,
            grow,
        )
        return new_penalty, tree_any(stalled)


class ToleranceSchedule(Protocol):
    """A rule that sets the inner solver's tolerance for each inner solve."""

    def initial_tolerance(self) -> float:
        """Return the tolerance for the first inner solve."""
        ...

    def update_tolerance(self, tolerance: Scalar) -> Scalar:
        """Return the tolerance for the next inner solve."""
        ...


class GeometricToleranceSchedule(eqx.Module):
    """A tolerance that starts at `initial` and shrinks by `factor` per solve.

    The solver never uses a tolerance below the `rtol` and `atol` of the inner
    solver, so the schedule ends at those values.
    """

    initial: float = 1e-2
    """The tolerance for the first inner solve."""
    factor: float = 0.1
    """The factor applied to the tolerance after each inner solve."""

    def initial_tolerance(self) -> float:
        """Return the tolerance for the first inner solve."""
        return self.initial

    def update_tolerance(self, tolerance: Scalar) -> Scalar:
        """Shrink the tolerance by `factor`."""
        return tolerance * self.factor
