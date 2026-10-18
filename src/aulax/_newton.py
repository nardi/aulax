from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from ._hessian import AutoHessian, HessianStrategy, PreparedHessian
from ._misc import tree_where


class _Prepared:
    """Holds a prepared Hessian in a static field.

    The prepared Hessian is created when the solver is traced and describes
    structure only, like a sparsity pattern, so it stays out of the traced state.
    The solver loop initializes the inner state more than once for the same problem,
    and the results describe the same structure, so all instances compare equal.
    """

    def __init__(self, value: PreparedHessian):
        self.value = value
        """The prepared Hessian."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Prepared)

    def __hash__(self) -> int:
        return 0


class _NewtonInfo(eqx.Module):
    """The value and gradient at the accepted point.

    The field names match the ones of the optimistix quasi-Newton minimisers,
    which is where `OptimistixStateAccess` looks for the gradient.
    """

    f: Scalar
    """The function value."""
    grad: PyTree[Float[Array, "..."]]
    """The gradient, shaped like `y`."""


class _NewtonState(eqx.Module):
    """The state of `Newton` between steps."""

    f_info: _NewtonInfo
    """The function value and gradient at the last accepted point."""
    hessian: Float[Array, "..."]
    """The data of the Hessian at the last accepted point."""
    first_step: Bool[Array, ""]
    """Whether the next evaluation is the first one, which is always accepted."""
    y_eval: PyTree[Float[Array, "..."]]
    """The trial point that the next step evaluates."""
    aux: PyTree
    """The aux output at the last accepted point."""
    radius: Scalar
    """The trust radius that bounds the next trial step."""
    terminate: Bool[Array, ""]
    """Whether the last step moved less than the tolerances allow."""
    result: optx.RESULTS
    """The outcome of the solve so far."""
    prepared: _Prepared = eqx.field(static=True)
    """The prepared Hessian, with the structure that the strategy detected."""


class Newton(optx.AbstractMinimiser):
    """Newton's method with the exact Hessian and a trust region.

    Each step evaluates the function and its gradient at a trial point. A trial
    point is accepted when the function decreased by at least `accept_ratio` times
    the decrease that the quadratic model predicted, and at an accepted point the
    Hessian is computed. The trust radius shrinks after a poor ratio, to the minimum
    of a quadratic fitted along the step, and grows after a good one that reached
    the boundary. Below the rounding level of the function value the ratio is noise,
    and a step that lowers the norm of the gradient is accepted.

    How the Hessian is represented and how the step inside the trust region is
    computed depends on `hessian`: dense linear algebra for small problems, a sparse
    factorization for large problems with sparse Hessians, and conjugate gradients
    with Hessian-vector products for the rest. The default chooses between them.

    The minimiser stops when the last trial step changed `y` and the function
    value by less than `rtol` and `atol` allow.
    """

    rtol: float = 1e-8
    """The relative tolerance of the stopping test."""
    atol: float = 1e-8
    """The absolute tolerance of the stopping test."""
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    """The norm in which the change of `y` is measured for the stopping test."""
    hessian: HessianStrategy = AutoHessian()
    """How the Hessian is represented and used."""
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
        prepared = self.hessian.prepare(fn, y, args)
        return _NewtonState(
            f_info=_NewtonInfo(
                f=jnp.zeros((), f_struct.dtype), grad=jax.tree.map(jnp.zeros_like, y)
            ),
            hessian=jnp.zeros(prepared.data_shape, flat_y.dtype),
            first_step=jnp.array(True),
            y_eval=y,
            aux=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), aux_struct
            ),
            radius=jnp.asarray(self.initial_radius, flat_y.dtype),
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
            prepared=_Prepared(prepared),
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
        prepared = state.prepared.value
        flat_eval, unravel = ravel_pytree(state.y_eval)

        def with_aux(flat: Array) -> tuple[Scalar, PyTree]:
            return fn(unravel(flat), args)

        def scalar(flat: Array) -> Scalar:
            return fn(unravel(flat), args)[0]

        if prepared.fused:
            f_eval, aux_eval, gradient_eval, hessian_eval = prepared.evaluate_fused(
                with_aux, flat_eval
            )
        else:
            (f_eval, aux_eval), gradient_eval = jax.value_and_grad(
                with_aux, has_aux=True
            )(flat_eval)

        flat_y, _ = ravel_pytree(y)
        trial_step = flat_eval - flat_y
        old = state.f_info
        old_gradient, _ = ravel_pytree(old.grad)
        slope = old_gradient @ trial_step
        model_curvature = trial_step @ prepared.matvec(
            scalar, state.hessian, trial_step
        )
        predicted = -(slope + 0.5 * model_curvature)
        ratio = (old.f - f_eval) / jnp.where(predicted > 0, predicted, jnp.inf)
        finite = jnp.isfinite(f_eval)
        # Below the rounding level of the function value, the ratio of actual to
        # predicted decrease is noise. A step that lowers the norm of the gradient is
        # then accepted, since otherwise the radius shrinks until the solve stalls.
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

        if prepared.fused:
            hessian = jnp.where(accept, hessian_eval, state.hessian)
        else:
            # The Hessian is only needed at points that are accepted.
            hessian = jax.lax.cond(
                accept,
                lambda: prepared.evaluate(scalar, flat_eval),
                lambda: state.hessian,
            )
        evaluated = _NewtonInfo(f=f_eval, grad=unravel(gradient_eval))
        f_info = tree_where(accept, evaluated, old)
        aux = tree_where(accept, aux_eval, state.aux)
        new_y = tree_where(accept, state.y_eval, y)

        flat_new_y, _ = ravel_pytree(new_y)
        direction = prepared.step(scalar, hessian, ravel_pytree(f_info.grad)[0], radius)
        y_scale = self.atol + self.rtol * jnp.abs(flat_eval)
        f_scale = self.atol + self.rtol * jnp.abs(f_eval)
        converged = (self.norm(trial_step / y_scale) < 1) & (
            jnp.abs(f_eval - old.f) < f_scale
        )
        new_state = _NewtonState(
            f_info=f_info,
            hessian=hessian,
            first_step=jnp.array(False),
            y_eval=unravel(flat_new_y + direction),
            aux=aux,
            radius=radius,
            terminate=converged & jnp.invert(state.first_step),
            result=optx.RESULTS.successful,
            prepared=state.prepared,
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
