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
from ._lbfgs import LBFGS
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
    $10^{-8}$ in float64 for `Newton` and `LBFGS`, which is the tolerance of IPOPT,
    and $10^{-6}$ in float64 for other minimisers. Float32 selects $10^{-5}$ for
    `Newton` and `LBFGS`, and $10^{-3}$ for other minimisers. The violation at
    the end of a solve is about the final inner tolerance of `inner`, so a bound
    much below that cannot be reached.
    """
    multiplier_update: MultiplierUpdate = SafeguardedMultiplierUpdate()
    """The rule that updates the multipliers after each inner solve."""
    penalty_update: PenaltyUpdate | None = None
    """The rule that sets and grows the penalty.

    `None` selects `AlgencanPenaltyUpdate()` for `Newton` and other minimisers.
    For `LBFGS` it selects an initial penalty of 10 and a growth factor of 100,
    since a quasi-Newton minimiser needs a larger penalty to reach feasibility in
    few outer iterations.
    """
    tolerance_schedule: ToleranceSchedule | None = None
    """The rule that sets the tolerance of each inner solve.

    `None` selects `GeometricToleranceSchedule()` for `Newton` and other
    minimisers, and a start at $10^{-8}$ for `LBFGS`.
    """
    inner_tolerance_floor: float | None = None
    r"""The smallest tolerance that an inner solve uses.

    The `rtol` and `atol` of `inner` are raised to this value. `None` selects zero
    in float64, so that the tolerances of `inner` apply, and $10^{-6}$ in float32,
    where gradients carry rounding errors of about that size and a tighter
    tolerance only adds steps.
    """
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
    inner_state_access: InnerStateAccess | None = None
    """Reads the evaluation and gradient at the accepted point of the inner solver.

    With the evaluation, the multipliers update as soon as an inner solve ends.
    Without it, the solver takes one more inner step to learn the constraint
    values at the inner solution. `None` selects `OptimistixStateAccess()`, which
    starts every inner solve without the Hessian approximation of the last one.
    For `LBFGS` it selects `OptimistixStateAccess` with
    `hessian_reuse="keep_unless_penalty_increased"`.
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
    penalty_freeze_factor: float | None = None
    r"""The penalty stops growing once the infeasibility is within this multiple of
    the tolerance.

    The infeasibility is the largest violation of feasibility or complementarity.
    Near the tolerance, the multipliers converge with the penalty held fixed, and
    a larger penalty only makes the inner problems harder. The multipliers keep
    updating. `None` selects 10 for `Newton` and 0.1 for other minimisers, since
    the quasi-Newton minimisers need many more steps with the penalty fixed. Zero
    turns the rule off.
    """
    gradient_noise_factor: float = 1e3
    r"""How far above rounding errors the tolerances on the gradient are kept.

    The gradient of the augmented Lagrangian contains the penalty times the
    constraint residuals, and so it carries rounding errors of about the machine
    epsilon times the penalty. The inner tolerance and `stationarity_tolerance` are
    raised to this factor times that, so that a solve with a large penalty can still
    end. Zero turns the rule off.
    """
    unbounded_threshold: float | None = None
    r"""The magnitude below which the augmented Lagrangian counts as unbounded.

    When the value of the augmented Lagrangian at an inner iterate is below minus
    this threshold, or the iterate is not finite, the inner solve has left the
    region where the penalty term bounds the objective. The solver then returns to
    the point where the inner solve started and grows the penalty. A line search
    can follow such a direction for many steps, while the trust region of `Newton`
    stops near the start. `None` selects $10^{20}$, the `fmin` of ALGENCAN, for
    minimisers other than `Newton`, and turns the check off for `Newton`.
    Infinity turns the check off.
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
        """Return `feasibility_tolerance`, or its default for `inner` and `dtype`."""
        if self.feasibility_tolerance is not None:
            return self.feasibility_tolerance
        if isinstance(self.inner, Newton | LBFGS):
            return dtype_default(dtype, 1e-8, 1e-5)
        return dtype_default(dtype, 1e-6, 1e-3)

    def penalty_update_for_inner(self) -> PenaltyUpdate:
        """Return `penalty_update`, or its default for `inner`."""
        if self.penalty_update is not None:
            return self.penalty_update
        if isinstance(self.inner, LBFGS):
            return AlgencanPenaltyUpdate(initial_value=10.0, growth_factor=100.0)
        return AlgencanPenaltyUpdate()

    def tolerance_schedule_for_inner(self) -> ToleranceSchedule:
        """Return `tolerance_schedule`, or its default for `inner`."""
        if self.tolerance_schedule is not None:
            return self.tolerance_schedule
        if isinstance(self.inner, LBFGS):
            return GeometricToleranceSchedule(initial=1e-8)
        return GeometricToleranceSchedule()

    def inner_state_access_for_inner(self) -> InnerStateAccess:
        """Return `inner_state_access`, or its default for `inner`."""
        if self.inner_state_access is not None:
            return self.inner_state_access
        if isinstance(self.inner, LBFGS):
            return OptimistixStateAccess(hessian_reuse="keep_unless_penalty_increased")
        return OptimistixStateAccess()

    def inner_tolerance_floor_for(self, dtype: DTypeLike) -> float:
        """Return `inner_tolerance_floor`, or its default for `dtype`."""
        if self.inner_tolerance_floor is not None:
            return self.inner_tolerance_floor
        return dtype_default(dtype, 0.0, 1e-6)

    def unbounded_threshold_for_inner(self) -> float:
        """Return `unbounded_threshold`, or its default for `inner`."""
        if self.unbounded_threshold is not None:
            return self.unbounded_threshold
        return math.inf if isinstance(self.inner, Newton) else 1e20

    def penalty_freeze_factor_for_inner(self) -> float:
        """Return `penalty_freeze_factor`, or its default for `inner`."""
        if self.penalty_freeze_factor is not None:
            return self.penalty_freeze_factor
        return 10.0 if isinstance(self.inner, Newton) else 0.1

    def gradient_noise_floor(self, penalty: Constraints, dtype: DTypeLike) -> Scalar:
        """Return the smallest tolerance on the gradient that rounding errors allow."""
        return (
            self.gradient_noise_factor * jnp.finfo(dtype).eps * tree_max_norm(penalty)
        )

    def final_inner_tolerance(self, dtype: DTypeLike) -> Scalar:
        """Return the tolerance at which the inner tolerance schedule ends."""
        return jnp.maximum(
            jnp.minimum(self.inner.rtol, self.inner.atol),
            self.inner_tolerance_floor_for(dtype),
        )

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

    def is_final_tolerance(
        self, tolerance: ArrayLike, dtype: DTypeLike
    ) -> Bool[ArrayLike, ""]:
        """Return whether `tolerance` has reached the end of the schedule."""
        return tolerance <= self.final_inner_tolerance(dtype)
