import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import ArrayLike, Bool, DTypeLike, Scalar

from ._misc import dtype_default
from ._updates import (
    AlgencanPenaltyUpdate,
    GeometricToleranceSchedule,
    MultiplierUpdate,
    PenaltyUpdate,
    SafeguardedMultiplierUpdate,
    ToleranceSchedule,
)


class AugmentedLagrangian(eqx.Module):
    """The augmented Lagrangian method around an optimistix minimiser.

    Each outer iteration minimizes the augmented Lagrangian with `inner` to a
    tolerance set by `tolerance_schedule`, then updates the multipliers and the
    penalty. The solve ends when the inner tolerance has reached the `rtol` and
    `atol` of `inner`, and the constraints hold to `feasibility_tolerance`.
    """

    inner: optx.AbstractMinimiser
    """The minimiser for the inner solves.

    Its `rtol` and `atol` are the tolerances of the last inner solve. Earlier
    inner solves use looser tolerances from `tolerance_schedule`.
    """
    feasibility_tolerance: float | None = None
    r"""The bound on constraint violation and complementarity at the solution.

    The solve stops when $\lVert h \rVert_\infty$ and
    $\lVert \min(-g, \mu) \rVert_\infty$ are both below it. `None` selects
    $10^{-6}$ in float64 and $10^{-3}$ in float32. A minimiser finds `y` to about
    the square root of the machine epsilon, so much tighter values make the
    penalty grow without improving the solution.
    """
    multiplier_update: MultiplierUpdate = SafeguardedMultiplierUpdate()
    """The rule that updates the multipliers after each inner solve."""
    penalty_update: PenaltyUpdate = AlgencanPenaltyUpdate()
    """The rule that sets and grows the penalty."""
    tolerance_schedule: ToleranceSchedule = GeometricToleranceSchedule()
    """The rule that sets the tolerance of each inner solve."""
    inner_max_steps: int = 256
    """The number of steps after which an inner solve counts as finished.

    Reaching it is not an error. The multipliers are updated from the last point,
    and the next inner solve continues from there.
    """

    def feasibility_tolerance_for(self, dtype: DTypeLike) -> float:
        """Return `feasibility_tolerance`, or its default for `dtype`."""
        if self.feasibility_tolerance is None:
            return dtype_default(dtype, 1e-6, 1e-3)
        return self.feasibility_tolerance

    def final_inner_tolerance(self) -> Scalar:
        """Return the tolerance at which the inner tolerance schedule ends."""
        return jnp.minimum(self.inner.rtol, self.inner.atol)

    def inner_at_tolerance(self, tolerance: ArrayLike) -> optx.AbstractMinimiser:
        """Return `inner` with its `rtol` and `atol` loosened to `tolerance`."""
        return eqx.tree_at(
            lambda inner: (inner.rtol, inner.atol),
            self.inner,
            (
                jnp.maximum(self.inner.rtol, tolerance),
                jnp.maximum(self.inner.atol, tolerance),
            ),
        )

    def is_final_tolerance(self, tolerance: ArrayLike) -> Bool[ArrayLike, ""]:
        """Return whether `tolerance` has reached the end of the schedule."""
        return tolerance <= self.final_inner_tolerance()
