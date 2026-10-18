import math

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import ArrayLike, Bool, DTypeLike, Scalar

from ._inner import (
    InnerStateAccess,
    InnerStoppingRule,
    OptimistixStateAccess,
)
from ._misc import dtype_default, tree_max_norm
from ._newton import Newton
from ._problem import Constraints
from ._scaling import GradientScaling, ProblemScaling
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

    inner: optx.AbstractMinimiser = Newton()
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
    stationarity_tolerance: float | None = None
    r"""The bound on the gradient of the Lagrangian at the solution.

    The solve only stops when the largest element of $\nabla_y L$ is at most this
    tolerance times the largest element of the objective gradient at `y0`, or
    one if that is smaller. This check needs the gradient from the inner solver,
    see `inner_state_access`. `None` selects $10^{-6}$ for the `Newton` minimiser,
    which converges that far, and turns the check off for other minimisers, whose
    stopping tests do not reach it. Infinity turns the check off.
    """
    inner_stopping: InnerStoppingRule | None = None
    """The rule that decides when an inner solve is finished.

    `None` uses the stopping test of the inner minimiser, and the gradient test when
    the minimiser takes inexact steps, which is the case for the matrix-free mode of
    `Newton`.
    """
    inner_state_access: InnerStateAccess = OptimistixStateAccess()
    """Reads the evaluation and gradient at the accepted point of the inner solver.

    With the evaluation, the multipliers update as soon as an inner solve ends.
    Without it, the solver takes one more inner step to learn the constraint
    values at the inner solution.
    """
    scaling: ProblemScaling | None = GradientScaling()
    """The rule that rescales the objective and constraints before the solve.

    `None` solves the problem as given. `GradientScaling` rescales by the gradients
    at `y0`, which helps when the constraints differ widely in magnitude. The
    feasibility tolerance, the multipliers and the statistics of the solution
    always refer to the original problem.
    """
    inner_max_steps: int = 256
    """The number of steps after which an inner solve counts as finished.

    Reaching it is not an error. The multipliers are updated from the last point,
    and the next inner solve continues from there.
    """
    penalty_freeze_fraction: float = 0.1
    r"""The penalty stops growing when the infeasibility is this fraction of its
    tolerance.

    The infeasibility is the largest violation of feasibility or complementarity.
    A problem that is feasible to well below `feasibility_tolerance` gains nothing
    from a larger penalty, and the penalty only makes the inner problems harder.
    The multipliers keep updating. Zero turns the rule off.
    """
    gradient_noise_factor: float = 1e3
    r"""How far above rounding errors the tolerances on the gradient are kept.

    The gradient of the augmented Lagrangian contains the penalty times the
    constraint residuals, and so it carries rounding errors of about the machine
    epsilon times the penalty. The inner tolerance and `stationarity_tolerance` are
    raised to this factor times that, so that a solve with a large penalty can still
    end. Zero turns the rule off.
    """
    verbose: bool = eqx.field(default=False, static=True)
    """Whether to print one line per outer iteration.

    The line shows the iteration, the number of inner steps, the objective, the
    infeasibility, the stationarity if the inner solver provides it, the largest
    penalty and the inner tolerance.
    """

    def stationarity_tolerance_for_inner(self) -> float:
        """Return `stationarity_tolerance`, or its default for the inner minimiser."""
        if self.stationarity_tolerance is not None:
            return self.stationarity_tolerance
        return 1e-6 if isinstance(self.inner, Newton) else math.inf

    def feasibility_tolerance_for(self, dtype: DTypeLike) -> float:
        """Return `feasibility_tolerance`, or its default for `dtype`."""
        if self.feasibility_tolerance is None:
            return dtype_default(dtype, 1e-6, 1e-3)
        return self.feasibility_tolerance

    def gradient_noise_floor(self, penalty: Constraints, dtype: DTypeLike) -> Scalar:
        """Return the smallest tolerance on the gradient that rounding errors allow."""
        return (
            self.gradient_noise_factor * jnp.finfo(dtype).eps * tree_max_norm(penalty)
        )

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
