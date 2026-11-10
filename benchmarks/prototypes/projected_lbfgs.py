"""A projected L-BFGS minimiser with a box, for experiments with aulax.

The search direction is the L-BFGS direction on the free variables, which are
those not held at a bound by the gradient (Kim, Sra and Dhillon, 2010). The line
search backtracks along the projected path with the Armijo test. The minimiser
works in the scaled variables `y / variable_scale`.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, Int, PyTree, Scalar


class _FunctionInfo(eqx.Module):
    """The objective value and its gradient at the last accepted point."""

    f: Scalar
    """Objective value at the accepted point."""
    grad: PyTree
    """Gradient of the objective, with the structure of `y`."""


class _ProjectedLBFGSState(eqx.Module):
    """The iteration state of `ProjectedLBFGS`."""

    f_info: _FunctionInfo
    """Objective value and gradient at the last accepted point."""
    aux: PyTree
    """Auxiliary output of `fn` at the last accepted point."""
    y_eval: PyTree
    """Point at which `fn` is evaluated next, with the structure of `y`."""
    direction: Float[Array, " n"]
    """Search direction in the scaled variables."""
    step_length: Scalar
    """Length of the step along `direction`."""
    first_step: Bool[Array, ""]
    """Whether the state is the initial one, before any step."""
    s_history: Float[Array, "m n"]
    """Ring buffer of secant steps, with the newest row at `history_index - 1`."""
    y_history: Float[Array, "m n"]
    """Ring buffer of secant gradient differences, aligned with `s_history`."""
    history_count: Int[Array, ""]
    """Number of valid rows in `s_history` and `y_history`."""
    history_index: Int[Array, ""]
    """Row of `s_history` and `y_history` that the next secant pair overwrites."""
    diagonal_numerator: Float[Array, " n"]
    """Decaying sum of step and gradient difference products for the diagonal fit."""
    diagonal_denominator: Float[Array, " n"]
    """Decaying sum of squared gradient differences, for the diagonal fit."""
    terminate: Bool[Array, ""]
    """Whether the minimiser has converged."""
    result: optx.RESULTS
    """Status of the last step."""


def _where(condition, true, false):
    """Select each leaf of `true` where `condition` holds, and of `false` elsewhere."""
    return jax.tree.map(
        lambda true_leaf, false_leaf: jnp.where(condition, true_leaf, false_leaf),
        true,
        false,
    )


class ProjectedLBFGS(optx.AbstractMinimiser):
    """Minimiser that projects L-BFGS steps onto the box of `lower` and `upper`."""

    rtol: float
    """Relative tolerance in the step and objective change tests."""
    atol: float
    """Absolute tolerance in the step, objective change and stationarity tests."""
    norm: Callable = optx.max_norm
    """Norm that measures the step relative to the tolerances."""
    lower: Array | None = None
    """Lower bounds on the flat `y`, with `-inf` for none."""
    upper: Array | None = None
    """Upper bounds on the flat `y`, with `inf` for none."""
    variable_scale: Array | None = None
    """Positive factors on the flat `y`. The minimiser works in `y / variable_scale`."""
    history_length: int = eqx.field(default=10, static=True)
    """Number of secant pairs kept for the L-BFGS inverse Hessian."""
    diagonal: bool = eqx.field(default=False, static=True)
    """Whether the initial inverse Hessian is a diagonal fitted to the secant pairs."""

    def _scale(self, flat_y):
        """Return the scale factors of the flat `y`, which are ones by default."""
        return (
            jnp.ones_like(flat_y)
            if self.variable_scale is None
            else self.variable_scale
        )

    def _box(self, flat_y):
        """Return the lower and upper bounds on the scaled flat `y`."""
        scale = self._scale(flat_y)
        lower = (
            jnp.full_like(flat_y, -jnp.inf)
            if self.lower is None
            else self.lower / scale
        )
        upper = (
            jnp.full_like(flat_y, jnp.inf) if self.upper is None else self.upper / scale
        )
        return lower, upper

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        """Return the initial state, with the point clipped into the box."""
        flat_y, unravel = ravel_pytree(y)
        scale = self._scale(flat_y)
        lower, upper = self._box(flat_y)
        scaled_point = jnp.clip(flat_y / scale, lower, upper)
        size = flat_y.size
        history_length = self.history_length
        return _ProjectedLBFGSState(
            f_info=_FunctionInfo(
                jnp.zeros((), f_struct.dtype), jax.tree.map(jnp.zeros_like, y)
            ),
            aux=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), aux_struct
            ),
            y_eval=unravel(scaled_point * scale),
            direction=jnp.zeros(size, flat_y.dtype),
            step_length=jnp.ones((), flat_y.dtype),
            first_step=jnp.array(True),
            s_history=jnp.zeros((history_length, size), flat_y.dtype),
            y_history=jnp.zeros((history_length, size), flat_y.dtype),
            history_count=jnp.array(0),
            history_index=jnp.array(0),
            diagonal_numerator=jnp.zeros(size, flat_y.dtype),
            diagonal_denominator=jnp.zeros(size, flat_y.dtype),
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
        )

    def _two_loop(self, state, gradient, initial):
        """Apply the L-BFGS inverse Hessian with initial diagonal `initial`."""
        history_length = self.history_length
        order = (state.history_index - 1 - jnp.arange(history_length)) % history_length
        valid = jnp.arange(history_length) < state.history_count

        def backward(vector, position):
            """Apply the backward loop update of one secant pair."""
            index = order[position]
            step_difference = state.s_history[index]
            gradient_difference = state.y_history[index]
            rho = 1.0 / jnp.where(
                valid[position], step_difference @ gradient_difference, 1.0
            )
            alpha = jnp.where(valid[position], rho * (step_difference @ vector), 0.0)
            return vector - alpha * gradient_difference, alpha

        vector, alphas = jax.lax.scan(backward, gradient, jnp.arange(history_length))
        result = initial * vector

        def forward(result, position):
            """Apply the forward loop update of one secant pair."""
            index = order[position]
            step_difference = state.s_history[index]
            gradient_difference = state.y_history[index]
            rho = 1.0 / jnp.where(
                valid[position], step_difference @ gradient_difference, 1.0
            )
            beta = jnp.where(valid[position], rho * (gradient_difference @ result), 0.0)
            correction = jnp.where(valid[position], alphas[position] - beta, 0.0)
            return result + step_difference * correction, None

        result, _ = jax.lax.scan(forward, result, jnp.arange(history_length)[::-1])
        return result

    def step(self, fn, y, args, options, state, tags):
        """Take one projected L-BFGS step and return the new `y`, state and `aux`."""
        flat_y, unravel = ravel_pytree(y)
        scale = self._scale(flat_y)
        lower, upper = self._box(flat_y)
        flat_trial, _ = ravel_pytree(state.y_eval)

        def scaled_function(scaled_point):
            """Evaluate `fn` at the unscaled point of `scaled_point`."""
            return fn(unravel(scaled_point * scale), args)

        (trial_value, trial_aux), trial_gradient = jax.value_and_grad(
            scaled_function, has_aux=True
        )(flat_trial / scale)
        scaled_point = flat_y / scale
        scaled_trial = flat_trial / scale
        old_value = state.f_info.f
        old_gradient = ravel_pytree(state.f_info.grad)[0] * scale
        displacement = scaled_trial - scaled_point
        decrease = old_gradient @ displacement
        finite = jnp.isfinite(trial_value) & jnp.all(jnp.isfinite(trial_gradient))
        armijo = trial_value <= old_value + 1e-4 * decrease

        def projected_gradient(point, gradient):
            """Return the projected gradient step at `point`, minus `point`."""
            return jnp.clip(point - gradient, lower, upper) - point

        noise = (
            100.0
            * jnp.finfo(trial_value.dtype).eps
            * jnp.maximum(1.0, jnp.abs(old_value))
        )
        below_noise = (jnp.abs(trial_value - old_value) < noise) & (
            jnp.max(jnp.abs(projected_gradient(scaled_trial, trial_gradient)))
            < jnp.max(jnp.abs(projected_gradient(scaled_point, old_gradient)))
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
        history_length = self.history_length
        s_history = jnp.where(
            useful,
            state.s_history.at[state.history_index].set(displacement),
            state.s_history,
        )
        y_history = jnp.where(
            useful,
            state.y_history.at[state.history_index].set(gradient_difference),
            state.y_history,
        )
        history_index = jnp.where(
            useful, (state.history_index + 1) % history_length, state.history_index
        )
        history_count = jnp.where(
            useful,
            jnp.minimum(state.history_count + 1, history_length),
            state.history_count,
        )
        diagonal_numerator = jnp.where(
            useful,
            0.9 * state.diagonal_numerator + displacement * gradient_difference,
            state.diagonal_numerator,
        )
        diagonal_denominator = jnp.where(
            useful,
            0.9 * state.diagonal_denominator + gradient_difference**2,
            state.diagonal_denominator,
        )

        new_scaled_point = jnp.where(accept, scaled_trial, scaled_point)
        new_value = jnp.where(accept, trial_value, old_value)
        new_gradient = jnp.where(accept, trial_gradient, old_gradient)

        # The variables held at a bound by the gradient stay fixed in the direction.
        bound_tolerance = jnp.minimum(
            1e-8, jnp.max(jnp.abs(projected_gradient(new_scaled_point, new_gradient)))
        )
        held = ((new_scaled_point <= lower + bound_tolerance) & (new_gradient > 0)) | (
            (new_scaled_point >= upper - bound_tolerance) & (new_gradient < 0)
        )
        free = jnp.invert(held)
        masked_gradient = jnp.where(free, new_gradient, 0.0)

        last_index = (history_index - 1) % history_length
        last_step_difference = s_history[last_index]
        last_gradient_difference = y_history[last_index]
        gamma = jnp.where(
            history_count > 0,
            (last_step_difference @ last_gradient_difference)
            / (last_gradient_difference @ last_gradient_difference),
            1.0,
        )
        if self.diagonal:
            fitted = diagonal_numerator / jnp.where(
                diagonal_denominator > 0, diagonal_denominator, 1.0
            )
            initial = jnp.where(
                (diagonal_numerator > 0) & (diagonal_denominator > 0),
                jnp.clip(fitted, 1e-6 * gamma, 1e6 * gamma),
                gamma,
            )
        else:
            initial = gamma * jnp.ones_like(new_scaled_point)

        def lbfgs_direction():
            """Return the L-BFGS direction on the free variables."""
            history = eqx.tree_at(
                lambda current: (
                    current.s_history,
                    current.y_history,
                    current.history_count,
                    current.history_index,
                ),
                state,
                (s_history, y_history, history_count, history_index),
            )
            return -jnp.where(
                free, self._two_loop(history, masked_gradient, initial), 0.0
            )

        direction = lbfgs_direction()
        gradient_size = jnp.max(jnp.abs(masked_gradient))
        steepest = -masked_gradient * jnp.where(
            history_count > 0, initial, 1.0 / jnp.maximum(1.0, gradient_size)
        )
        descent = (direction @ new_gradient < 0) & jnp.all(jnp.isfinite(direction))
        direction = jnp.where(descent & (history_count > 0), direction, steepest)

        # After a rejected trial, shrink the step to the minimum of a quadratic fitted
        # to the value, the slope and the trial value along the step.
        slope = state.direction @ old_gradient * state.step_length
        gap = trial_value - old_value - slope
        interpolated = -slope / jnp.where(gap > 0, 2.0 * gap, jnp.inf)
        shrink = jnp.where(
            finite & jnp.isfinite(interpolated),
            jnp.clip(interpolated, 0.1, 0.5),
            0.1,
        )
        shrunk_length = state.step_length * shrink
        stalled = jnp.invert(accept) & (shrunk_length < 1e-12)
        # A stalled line search restarts from a steepest descent step without history.
        restart = stalled & (state.history_count > 0)
        history_count = jnp.where(restart, 0, history_count)
        direction = jnp.where(
            accept, direction, jnp.where(restart, steepest, state.direction)
        )
        step_length = jnp.where(accept | restart, 1.0, shrunk_length)
        next_scaled_point = jnp.clip(
            new_scaled_point + step_length * direction, lower, upper
        )

        y_scale = self.atol + self.rtol * jnp.abs(scaled_trial)
        f_scale = self.atol + self.rtol * jnp.abs(trial_value)
        small_step = (self.norm(displacement / y_scale) < 1) & (
            jnp.abs(trial_value - old_value) < f_scale
        )
        full_step = state.step_length >= 1.0
        stationary = (
            jnp.max(jnp.abs(projected_gradient(new_scaled_point, new_gradient)))
            <= self.atol
        )
        converged = jnp.invert(state.first_step) & (
            (accept & ((small_step & full_step) | stationary))
            | (stalled & jnp.invert(restart))
        )

        aux = _where(accept, trial_aux, state.aux)
        new_y = unravel(new_scaled_point * scale)
        new_state = _ProjectedLBFGSState(
            f_info=_FunctionInfo(new_value, unravel(new_gradient / scale)),
            aux=aux,
            y_eval=unravel(next_scaled_point * scale),
            direction=direction,
            step_length=step_length,
            first_step=jnp.array(False),
            s_history=s_history,
            y_history=y_history,
            history_count=history_count,
            history_index=history_index,
            diagonal_numerator=diagonal_numerator,
            diagonal_denominator=diagonal_denominator,
            terminate=converged,
            result=optx.RESULTS.successful,
        )
        input_aux = _where(state.first_step, aux, state.aux)
        return new_y, new_state, input_aux

    def terminate(self, fn, y, args, options, state, tags):
        """Return the termination flag of `state` and its result status."""
        return state.terminate, state.result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        """Return `y` and `aux` unchanged, with no extra statistics."""
        return y, aux, {}
