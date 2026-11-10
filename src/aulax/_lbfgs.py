from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, Int, PyTree, Scalar

from ._bounds import Bounds
from ._misc import tree_where


VARIABLE_SCALING_OPTIONS = ("bounds", "none")
"""The choices for `LBFGS.variable_scaling`."""


class _LBFGSInfo(eqx.Module):
    """The value and gradient at the accepted point.

    The field names match the ones of the optimistix quasi-Newton minimisers,
    which is where `OptimistixStateAccess` looks for the gradient.
    """

    f: Scalar
    """The function value."""
    grad: PyTree[Float[Array, "..."]]
    """The gradient of the function in the original variables, shaped like `y`."""


class _LBFGSHistory(eqx.Module):
    """The secant pairs of the inverse Hessian approximation.

    The pairs are stored in the scaled variables, as rows of two ring buffers.
    """

    steps: Float[Array, "history size"]
    """The steps between accepted points, with the newest at `index - 1`."""
    gradient_differences: Float[Array, "history size"]
    """The differences of the gradient over the steps, in the rows of `steps`."""
    count: Int[Array, ""]
    """The number of rows that hold a secant pair."""
    index: Int[Array, ""]
    """The row that the next secant pair overwrites."""
    diagonal_numerator: Float[Array, " size"] | None
    """The decaying sum of products of step and gradient difference, per variable.

    This is `None` unless `LBFGS.diagonal` is set.
    """
    diagonal_denominator: Float[Array, " size"] | None
    """The decaying sum of squared gradient differences, per variable.

    This is `None` unless `LBFGS.diagonal` is set.
    """


class _LBFGSState(eqx.Module):
    """The state of `LBFGS` between steps."""

    f_info: _LBFGSInfo
    """The function value and gradient at the last accepted point."""
    aux: PyTree
    """The aux output at the last accepted point."""
    y_eval: PyTree[Float[Array, "..."]]
    """The trial point that the next step evaluates."""
    direction: Float[Array, " size"]
    """The search direction in the scaled variables."""
    step_length: Scalar
    """The multiple of `direction` that leads to `y_eval`."""
    first_step: Bool[Array, ""]
    """Whether the next evaluation is the first one, which is always accepted."""
    history: _LBFGSHistory
    """The secant pairs that define the inverse Hessian approximation."""
    variable_scale: Float[Array, " size"] | None
    """The factors that divide the variables, or `None` if all are one."""
    terminate: Bool[Array, ""]
    """Whether the minimiser has converged."""
    result: optx.RESULTS
    """The outcome of the solve so far."""


class LBFGS(optx.AbstractMinimiser):
    r"""Limited-memory BFGS with an Armijo line search and optional bounds.

    The search direction is the product of an inverse Hessian approximation from
    the last `history_length` secant pairs with the negative gradient. Each step
    evaluates the function and its gradient at one trial point. The trial point is
    accepted when the function decreased by at least $10^{-4}$ times the decrease
    that the gradient predicts along the step. After a rejection, the step length
    shrinks to the minimum of a quadratic fitted along the step. A function value
    that changes by less than its rounding level also counts as accepted when the
    projected gradient decreases. After a line search that cannot find a decrease,
    the history is discarded and the next step is a steepest descent step.

    With `options["bounds"]`, an expanded `Bounds`, every trial point is clipped
    into the box, so that no iterate leaves it. The direction is then computed on
    the free variables only, which are the ones that the gradient does not hold at
    a bound. Without bounds, the clipping and the masking are not traced and the
    code is the one of an unconstrained minimiser.

    The minimiser stops when the last accepted step changed the variables and the
    function value by less than `rtol` and `atol` allow, or when the projected
    gradient in the scaled variables is at most `atol`. Both tests refer to the
    variables divided by the scale that `variable_scaling` selects.
    """

    rtol: float = 1e-8
    """The relative tolerance of the stopping test."""
    atol: float = 1e-8
    """The absolute tolerance of the stopping tests."""
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    """The norm in which the change of the variables is measured."""
    history_length: int = eqx.field(default=10, static=True)
    """The number of secant pairs in the inverse Hessian approximation."""
    variable_scaling: str = eqx.field(default="bounds", static=True)
    """How the variables are scaled, which is `"bounds"` or `"none"`.

    With `"bounds"`, a variable with two finite bounds is divided by the width of
    its box, and the others by one. The scale is computed from `options["bounds"]`
    when the state is created. `"none"` divides by one everywhere.
    """
    diagonal: bool = eqx.field(default=False, static=True)
    """Whether the initial inverse Hessian is a diagonal fitted to the secant pairs.

    Otherwise the initial inverse Hessian is a multiple of the identity.
    """

    supports_bounds: ClassVar[bool] = True
    """Whether the minimiser reads `options["bounds"]` and keeps iterates inside it."""

    def __check_init__(self):
        if self.variable_scaling not in VARIABLE_SCALING_OPTIONS:
            raise ValueError(
                f"`variable_scaling` must be one of {VARIABLE_SCALING_OPTIONS}, "
                f"not {self.variable_scaling!r}."
            )

    def _flat_bounds(
        self, y: PyTree[Float[Array, "..."]], options: dict[str, Any]
    ) -> tuple[Float[Array, " size"], Float[Array, " size"]] | None:
        """Return the flat lower and upper bounds, or `None` without bounds."""
        bounds: Bounds | None = options.get("bounds")
        if bounds is None:
            return None
        expanded = bounds.expand(y)
        return ravel_pytree(expanded.lower)[0], ravel_pytree(expanded.upper)[0]

    def _variable_scale(
        self, flat_bounds: tuple[Array, Array] | None
    ) -> Float[Array, " size"] | None:
        """Return the box widths as scale factors, with one for unbounded variables."""
        if flat_bounds is None or self.variable_scaling == "none":
            return None
        lower, upper = flat_bounds
        width = upper - lower
        usable = jnp.isfinite(width) & (width > 0)
        return jnp.where(usable, width, 1.0)

    def init(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _LBFGSState:
        """Return a state whose first step evaluates the function at `y` in the box."""
        flat_y, unravel = ravel_pytree(y)
        flat_bounds = self._flat_bounds(y, options)
        if flat_bounds is not None:
            flat_y = jnp.clip(flat_y, *flat_bounds)
        size = flat_y.size
        diagonal = jnp.zeros(size, flat_y.dtype) if self.diagonal else None
        return _LBFGSState(
            f_info=_LBFGSInfo(
                f=jnp.zeros((), f_struct.dtype), grad=jax.tree.map(jnp.zeros_like, y)
            ),
            aux=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), aux_struct
            ),
            y_eval=unravel(flat_y),
            direction=jnp.zeros(size, flat_y.dtype),
            step_length=jnp.ones((), flat_y.dtype),
            first_step=jnp.array(True),
            history=_LBFGSHistory(
                steps=jnp.zeros((self.history_length, size), flat_y.dtype),
                gradient_differences=jnp.zeros(
                    (self.history_length, size), flat_y.dtype
                ),
                count=jnp.array(0),
                index=jnp.array(0),
                diagonal_numerator=diagonal,
                diagonal_denominator=diagonal,
            ),
            variable_scale=self._variable_scale(flat_bounds),
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
        )

    def _inverse_hessian_product(
        self,
        history: _LBFGSHistory,
        vector: Float[Array, " size"],
        initial_inverse: Float[Array, " size"],
    ) -> Float[Array, " size"]:
        """Return the product of the inverse Hessian approximation with `vector`.

        This is the two-loop recursion of Nocedal. `initial_inverse` is the
        diagonal of the inverse Hessian that the recursion starts from.
        """
        length = self.history_length
        # The rows are visited from the newest to the oldest, and unused rows are
        # skipped by masking, so that the loop has a fixed length.
        order = (history.index - 1 - jnp.arange(length)) % length
        valid = jnp.arange(length) < history.count
        steps = history.steps[order]
        differences = history.gradient_differences[order]
        curvature = jnp.sum(steps * differences, axis=1)
        inverse_curvature = jnp.where(
            valid, 1.0 / jnp.where(valid, curvature, 1.0), 0.0
        )

        def backward(
            remaining: Float[Array, " size"], pair: tuple[Array, Array, Array]
        ) -> tuple[Float[Array, " size"], Scalar]:
            step, difference, weight = pair
            coefficient = weight * (step @ remaining)
            return remaining - coefficient * difference, coefficient

        remaining, coefficients = jax.lax.scan(
            backward, vector, (steps, differences, inverse_curvature)
        )

        def forward(
            result: Float[Array, " size"], pair: tuple[Array, Array, Array, Scalar]
        ) -> tuple[Float[Array, " size"], None]:
            step, difference, weight, coefficient = pair
            correction = coefficient - weight * (difference @ result)
            return result + correction * step, None

        result, _ = jax.lax.scan(
            forward,
            initial_inverse * remaining,
            (steps, differences, inverse_curvature, coefficients),
            reverse=True,
        )
        return result

    def step(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        state: _LBFGSState,
        tags: frozenset[object],
    ) -> tuple[PyTree[Float[Array, "..."]], _LBFGSState, PyTree]:
        """Evaluate the trial point, accept or reject it, and plan the next trial.

        Like the optimistix minimisers, this returns the aux output of the point
        that went into the step, and keeps the one of the accepted point in the
        state.
        """
        flat_y, unravel = ravel_pytree(y)
        flat_trial, _ = ravel_pytree(state.y_eval)
        flat_bounds = self._flat_bounds(y, options)
        scale = state.variable_scale
        history = state.history

        def to_scaled(flat: Array) -> Array:
            return flat if scale is None else flat / scale

        def to_original(scaled: Array) -> Array:
            """Return the variables for a scaled point, inside the box."""
            flat = scaled if scale is None else scaled * scale
            return flat if flat_bounds is None else jnp.clip(flat, *flat_bounds)

        scaled_box = (
            None
            if flat_bounds is None
            else (to_scaled(flat_bounds[0]), to_scaled(flat_bounds[1]))
        )

        def projected_gradient_size(point: Array, gradient: Array) -> Scalar:
            """Return the largest element of the step to the projected gradient step."""
            if scaled_box is None:
                return jnp.max(jnp.abs(gradient))
            step_to_box = jnp.clip(point - gradient, *scaled_box)
            return jnp.max(jnp.abs(step_to_box - point))

        def scaled_function(scaled_point: Array) -> tuple[Scalar, PyTree]:
            return fn(unravel(to_original(scaled_point)), args)

        scaled_trial = to_scaled(flat_trial)
        (trial_value, trial_aux), trial_gradient = jax.value_and_grad(
            scaled_function, has_aux=True
        )(scaled_trial)
        scaled_point = to_scaled(flat_y)
        old = state.f_info
        old_value = old.f
        old_gradient = ravel_pytree(old.grad)[0]
        if scale is not None:
            old_gradient = old_gradient * scale
        displacement = scaled_trial - scaled_point
        predicted_change = old_gradient @ displacement
        finite = jnp.isfinite(trial_value) & jnp.all(jnp.isfinite(trial_gradient))
        armijo = trial_value <= old_value + 1e-4 * predicted_change
        # Below the rounding level of the function value, the Armijo test is noise.
        # A step that lowers the projected gradient is then accepted, since
        # otherwise the step length shrinks until the solve stalls.
        noise = (
            100.0
            * jnp.finfo(trial_value.dtype).eps
            * jnp.maximum(1.0, jnp.abs(old_value))
        )
        below_noise = (jnp.abs(trial_value - old_value) < noise) & (
            projected_gradient_size(scaled_trial, trial_gradient)
            < projected_gradient_size(scaled_point, old_gradient)
        )
        accept = state.first_step | ((armijo | below_noise) & finite)

        # The secant pair of an accepted step joins the history when its curvature
        # is positive.
        gradient_difference = trial_gradient - old_gradient
        curvature = displacement @ gradient_difference
        useful = (
            accept
            & jnp.invert(state.first_step)
            & (
                curvature
                > 1e-10
                * jnp.linalg.norm(displacement)
                * jnp.linalg.norm(gradient_difference)
            )
        )
        length = self.history_length
        steps = jnp.where(
            useful,
            history.steps.at[history.index].set(displacement),
            history.steps,
        )
        gradient_differences = jnp.where(
            useful,
            history.gradient_differences.at[history.index].set(gradient_difference),
            history.gradient_differences,
        )
        index = jnp.where(useful, (history.index + 1) % length, history.index)
        count = jnp.where(useful, jnp.minimum(history.count + 1, length), history.count)
        if self.diagonal:
            assert history.diagonal_numerator is not None
            assert history.diagonal_denominator is not None
            diagonal_numerator = jnp.where(
                useful,
                0.9 * history.diagonal_numerator + displacement * gradient_difference,
                history.diagonal_numerator,
            )
            diagonal_denominator = jnp.where(
                useful,
                0.9 * history.diagonal_denominator + gradient_difference**2,
                history.diagonal_denominator,
            )
        else:
            diagonal_numerator = diagonal_denominator = None
        new_history = _LBFGSHistory(
            steps=steps,
            gradient_differences=gradient_differences,
            count=count,
            index=index,
            diagonal_numerator=diagonal_numerator,
            diagonal_denominator=diagonal_denominator,
        )

        new_scaled_point = jnp.where(accept, scaled_trial, scaled_point)
        new_value = jnp.where(accept, trial_value, old_value)
        new_gradient = jnp.where(accept, trial_gradient, old_gradient)

        if scaled_box is None:
            free = None
            masked_gradient = new_gradient
        else:
            scaled_lower, scaled_upper = scaled_box
            # A variable at a bound is held there when the gradient points out of
            # the box. The tolerance also holds variables that are very close.
            bound_tolerance = jnp.minimum(
                1e-8, projected_gradient_size(new_scaled_point, new_gradient)
            )
            held = (
                (new_scaled_point <= scaled_lower + bound_tolerance)
                & (new_gradient > 0)
            ) | (
                (new_scaled_point >= scaled_upper - bound_tolerance)
                & (new_gradient < 0)
            )
            free = jnp.invert(held)
            masked_gradient = jnp.where(free, new_gradient, 0.0)

        last = (index - 1) % length
        last_step = steps[last]
        last_difference = gradient_differences[last]
        gamma = jnp.where(
            count > 0,
            (last_step @ last_difference) / (last_difference @ last_difference),
            1.0,
        )
        if self.diagonal:
            assert diagonal_numerator is not None
            assert diagonal_denominator is not None
            fitted = diagonal_numerator / jnp.where(
                diagonal_denominator > 0, diagonal_denominator, 1.0
            )
            initial_inverse = jnp.where(
                (diagonal_numerator > 0) & (diagonal_denominator > 0),
                jnp.clip(fitted, 1e-6 * gamma, 1e6 * gamma),
                gamma,
            )
        else:
            initial_inverse = gamma * jnp.ones_like(new_scaled_point)

        quasi_newton = -self._inverse_hessian_product(
            new_history, masked_gradient, initial_inverse
        )
        if free is not None:
            quasi_newton = jnp.where(free, quasi_newton, 0.0)
        gradient_size = jnp.max(jnp.abs(masked_gradient))
        steepest = -masked_gradient * jnp.where(
            count > 0, initial_inverse, 1.0 / jnp.maximum(1.0, gradient_size)
        )
        descent = (quasi_newton @ new_gradient < 0) & jnp.all(
            jnp.isfinite(quasi_newton)
        )
        direction = jnp.where(descent & (count > 0), quasi_newton, steepest)

        # After a rejected trial, shrink the step to the minimum of a quadratic
        # fitted to the value, the slope and the trial value along the step.
        curvature_gap = trial_value - old_value - predicted_change
        interpolated = -predicted_change / jnp.where(
            curvature_gap > 0, 2.0 * curvature_gap, jnp.inf
        )
        shrink = jnp.where(
            finite & jnp.isfinite(interpolated),
            jnp.clip(interpolated, 0.1, 0.5),
            0.1,
        )
        shrunk_length = state.step_length * shrink
        smallest_length = jnp.finfo(trial_value.dtype).eps ** 0.75
        stalled = jnp.invert(accept) & (shrunk_length < smallest_length)
        # A stalled line search restarts from a steepest descent step without
        # history, and the minimiser stops if it had no history already.
        restart = stalled & (state.history.count > 0)
        new_history = eqx.tree_at(
            lambda current: current.count,
            new_history,
            jnp.where(restart, 0, new_history.count),
        )
        direction = jnp.where(
            accept, direction, jnp.where(restart, steepest, state.direction)
        )
        step_length = jnp.where(accept | restart, 1.0, shrunk_length)
        next_flat = to_original(new_scaled_point + step_length * direction)

        y_scale = self.atol + self.rtol * jnp.abs(scaled_trial)
        f_scale = self.atol + self.rtol * jnp.abs(trial_value)
        small_step = (self.norm(displacement / y_scale) < 1) & (
            jnp.abs(trial_value - old_value) < f_scale
        )
        full_step = state.step_length >= 1.0
        stationary = (
            projected_gradient_size(new_scaled_point, new_gradient) <= self.atol
        )
        converged = jnp.invert(state.first_step) & (
            (accept & ((small_step & full_step) | stationary))
            | (stalled & jnp.invert(restart))
        )

        accepted_gradient = new_gradient if scale is None else new_gradient / scale
        aux = tree_where(accept, trial_aux, state.aux)
        new_y = unravel(to_original(new_scaled_point))
        new_state = _LBFGSState(
            f_info=_LBFGSInfo(f=new_value, grad=unravel(accepted_gradient)),
            aux=aux,
            y_eval=unravel(next_flat),
            direction=direction,
            step_length=step_length,
            first_step=jnp.array(False),
            history=new_history,
            variable_scale=scale,
            terminate=converged,
            result=optx.RESULTS.successful,
        )
        input_aux = tree_where(state.first_step, aux, state.aux)
        return new_y, new_state, input_aux

    def terminate(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        state: _LBFGSState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        """Return whether the minimiser has converged."""
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        aux: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: _LBFGSState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[PyTree[Float[Array, "..."]], PyTree, dict[str, Any]]:
        """Return the solution unchanged."""
        return y, aux, {}
