from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from ._misc import tree_where


class _NewtonInfo(eqx.Module):
    """The value, gradient and Hessian of the function at the accepted point.

    The field names match the ones of the optimistix quasi-Newton minimisers,
    which is where `OptimistixStateAccess` looks for the gradient.
    """

    f: Scalar
    """The function value."""
    grad: PyTree[Float[Array, "..."]]
    """The gradient, shaped like `y`."""
    hessian: lx.MatrixLinearOperator
    """The Hessian, as a dense matrix over the flattened `y`."""


class _NewtonState(eqx.Module):
    """The state of `NewtonTrustRegion` between steps."""

    first_step: Bool[Array, ""]
    """Whether the next evaluation is the first one, which is always accepted."""
    y_eval: PyTree[Float[Array, "..."]]
    """The trial point that the next step evaluates."""
    f_info: _NewtonInfo
    """The function value, gradient and Hessian at the last accepted point."""
    aux: PyTree
    """The aux output at the last accepted point."""
    radius: Scalar
    """The trust radius that bounds the next trial step."""
    terminate: Bool[Array, ""]
    """Whether the last step moved less than the tolerances allow."""
    result: optx.RESULTS
    """The outcome of the solve so far."""


def trust_region_step(
    hessian: Float[Array, "n n"], gradient: Float[Array, " n"], radius: Scalar
) -> Float[Array, " n"]:
    r"""Return the minimizer of the quadratic model within the trust region.

    The model is $g^\top p + \frac{1}{2} p^\top H p$ with
    $\lVert p \rVert_2 \le \Delta$. The step is the Newton step when $H$ is
    positive definite and the step fits in the region. Otherwise it is
    $-(H + \sigma I)^{-1} g$ with the shift $\sigma$ that makes $H + \sigma I$
    positive definite and the step as long as the radius. An eigendecomposition
    of $H$ reduces the search for $\sigma$ to a one-dimensional bisection. The
    case where the gradient is orthogonal to the eigenvector of the smallest
    eigenvalue is not treated separately, and the step then stops short of the
    boundary.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    projected = eigenvectors.T @ gradient
    largest = jnp.max(jnp.abs(eigenvalues))
    # Eigenvalues below this are rounding noise, and count as zero curvature.
    floor = jnp.finfo(hessian.dtype).eps * largest + jnp.finfo(hessian.dtype).tiny

    def step_length(shift: Scalar) -> Scalar:
        return jnp.linalg.norm(projected / (eigenvalues + shift))

    positive_definite = eigenvalues[0] > floor
    newton_fits = positive_definite & (step_length(0.0) <= radius)
    lower = jnp.maximum(0.0, -eigenvalues[0]) + floor
    upper = lower + jnp.linalg.norm(gradient) / radius + largest

    def bisect(_, bounds: tuple[Scalar, Scalar]) -> tuple[Scalar, Scalar]:
        low, high = bounds
        middle = 0.5 * (low + high)
        too_long = step_length(middle) > radius
        return jnp.where(too_long, middle, low), jnp.where(too_long, high, middle)

    low, high = jax.lax.fori_loop(0, 100, bisect, (lower, upper))
    shift = jnp.where(newton_fits, 0.0, 0.5 * (low + high))
    return -eigenvectors @ (projected / (eigenvalues + shift))


class NewtonTrustRegion(optx.AbstractMinimiser):
    """Newton's method with the exact Hessian and a trust region.

    Each step evaluates the function, its gradient and its dense Hessian at the
    trial point in one pass, using forward-mode differentiation of the reverse-mode
    gradient. A trial point is accepted when the function decreased by at least
    `accept_ratio` times the decrease that the quadratic model predicted. The
    trust radius shrinks after a poor ratio and grows after a good one that
    reached the boundary.

    The Hessian is dense and costs one eigendecomposition per accepted step, so
    this suits problems with up to a few hundred variables. It handles
    indefinite Hessians through the trust region, and ill-conditioned ones
    like those of an augmented Lagrangian with a large penalty, because the
    Newton step is exact in the directions the penalty dominates.

    The minimiser stops when the last trial step changed `y` and the function
    value by less than `rtol` and `atol` allow.
    """

    rtol: float
    """The relative tolerance of the stopping test."""
    atol: float
    """The absolute tolerance of the stopping test."""
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    """The norm in which the change of `y` is measured for the stopping test."""
    initial_radius: float = 10.0
    """The trust radius of the first step."""
    maximum_radius: float = 1e8
    """The largest trust radius."""
    accept_ratio: float = 1e-4
    """The smallest ratio of actual to predicted decrease that accepts a step."""

    def init(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NewtonState:
        """Return a state whose first step evaluates the function at `y`."""
        flat_y, _ = ravel_pytree(y)
        size = flat_y.size
        f_info = _NewtonInfo(
            f=jnp.zeros((), f_struct.dtype),
            grad=jax.tree.map(jnp.zeros_like, y),
            hessian=lx.MatrixLinearOperator(jnp.zeros((size, size), flat_y.dtype)),
        )
        return _NewtonState(
            first_step=jnp.array(True),
            y_eval=y,
            f_info=f_info,
            aux=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), aux_struct
            ),
            radius=jnp.asarray(self.initial_radius, flat_y.dtype),
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
        )

    def step(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonState,
        tags: frozenset[object],
    ) -> tuple[PyTree[Float[Array, "..."]], _NewtonState, PyTree]:
        """Evaluate the trial point, accept or reject it, and plan the next trial.

        Like the optimistix minimisers, this returns the aux output of the point
        that went into the step, and keeps the one of the accepted point in the
        state.
        """
        flat_eval, unravel = ravel_pytree(state.y_eval)

        def gradient_with_values(flat_y: Float[Array, " n"]):
            (value, aux), gradient = jax.value_and_grad(
                lambda point: fn(point, args), has_aux=True
            )(unravel(flat_y))
            flat_gradient, _ = ravel_pytree(gradient)
            return flat_gradient, (value, aux, flat_gradient)

        # One pass gives the value, the gradient and the Hessian, so that a trial
        # point costs one evaluation of the function.
        hessian_eval, (f_eval, aux_eval, gradient_eval) = jax.jacfwd(
            gradient_with_values, has_aux=True
        )(flat_eval)
        hessian_eval = 0.5 * (hessian_eval + hessian_eval.T)

        flat_y, _ = ravel_pytree(y)
        trial_step = flat_eval - flat_y
        old = state.f_info
        old_gradient, _ = ravel_pytree(old.grad)
        predicted = -(
            old_gradient @ trial_step
            + 0.5 * trial_step @ (old.hessian.as_matrix() @ trial_step)
        )
        ratio = (old.f - f_eval) / jnp.where(predicted > 0, predicted, jnp.inf)
        finite = jnp.isfinite(f_eval)
        # Below the rounding level of the function value, the ratio of actual to
        # predicted decrease is noise. A step that lowers the norm of the gradient
        # is then accepted, since otherwise the radius shrinks until the solve stalls.
        noise = 100.0 * jnp.finfo(f_eval.dtype).eps * jnp.maximum(1.0, jnp.abs(old.f))
        below_noise = (jnp.abs(predicted) < noise) & (jnp.abs(f_eval - old.f) < noise)
        gradient_decreased = jnp.linalg.norm(gradient_eval) < jnp.linalg.norm(
            old_gradient
        )
        accept = state.first_step | (
            ((ratio > self.accept_ratio) | (below_noise & gradient_decreased)) & finite
        )

        trial_length = jnp.linalg.norm(trial_step)
        # After a poor step, shrink to where a quadratic fitted to the value, the
        # slope and the new value along the step has its minimum. A penalty term
        # that switches on inside the step makes the model far too optimistic,
        # and shrinking by a fixed factor then takes many evaluations.
        slope = old_gradient @ trial_step
        curvature_gap = f_eval - old.f - slope
        interpolated = -slope / jnp.where(
            curvature_gap > 0, 2.0 * curvature_gap, jnp.inf
        )
        shrink = jnp.where(
            jnp.isfinite(interpolated), jnp.clip(interpolated, 1e-2, 0.25), 0.25
        )
        grown = jnp.where(
            (ratio > 0.75) & (trial_length >= 0.99 * state.radius),
            jnp.minimum(2.0 * state.radius, self.maximum_radius),
            state.radius,
        )
        radius = jnp.where(ratio < 0.25, shrink * trial_length, grown)
        radius = jnp.where(finite, radius, 0.25 * trial_length)
        radius = jnp.where(state.first_step, state.radius, radius)

        evaluated = _NewtonInfo(
            f=f_eval,
            grad=unravel(gradient_eval),
            hessian=lx.MatrixLinearOperator(hessian_eval),
        )
        f_info = tree_where(accept, evaluated, old)
        aux = tree_where(accept, aux_eval, state.aux)
        new_y = tree_where(accept, state.y_eval, y)

        flat_new_y, _ = ravel_pytree(new_y)
        direction = trust_region_step(
            f_info.hessian.as_matrix(), ravel_pytree(f_info.grad)[0], radius
        )
        y_scale = self.atol + self.rtol * jnp.abs(flat_eval)
        f_scale = self.atol + self.rtol * jnp.abs(f_eval)
        converged = (self.norm(trial_step / y_scale) < 1) & (
            jnp.abs(f_eval - old.f) < f_scale
        )
        new_state = _NewtonState(
            first_step=jnp.array(False),
            y_eval=unravel(flat_new_y + direction),
            f_info=f_info,
            aux=aux,
            radius=radius,
            terminate=converged & jnp.invert(state.first_step),
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
        state: _NewtonState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        """Return whether the last step moved less than the tolerances allow."""
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        aux: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[PyTree[Float[Array, "..."]], PyTree, dict[str, Any]]:
        """Return the solution unchanged."""
        return y, aux, {}
