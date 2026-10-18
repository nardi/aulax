from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, PyTree, Scalar

from ._misc import tree_max_norm, tree_where
from ._problem import Evaluation


class InnerStateAccess(Protocol):
    """Access to the parts of an inner solver state that the outer loop can use.

    The state of an optimistix minimiser is private to optimistix. This protocol
    keeps the solver loop independent of its layout. A method returns `None`
    when the state does not have what it asks for, and the loop then falls back
    to what it can do without.
    """

    def accepted_evaluation(self, inner_state: PyTree) -> Evaluation | None:
        """Return the evaluation at the last accepted point, or `None`."""
        ...

    def accepted_gradient(self, inner_state: PyTree) -> PyTree[Array] | None:
        """Return the gradient of the augmented Lagrangian there, or `None`."""
        ...

    def prefers_gradient_stopping(self, inner_state: PyTree) -> bool:
        """Return whether the inner solver's own stopping test is unreliable.

        A minimiser that takes inexact steps, such as truncated conjugate gradients,
        can make a small step while the gradient is still large. The solver then
        stops on the gradient when no stopping rule was chosen.
        """
        ...

    def transfer_state(
        self,
        old_state: PyTree,
        new_state: PyTree,
        penalty_ratio: Scalar,
    ) -> PyTree:
        """Return the inner state to continue with after an outer update.

        `new_state` is a fresh state for the new augmented Lagrangian, and
        `old_state` is the state at the end of the previous inner solve. The
        result may carry information over from `old_state`, such as a Hessian
        approximation. `penalty_ratio` is the largest ratio of a new to an old
        penalty, which is one when the penalty stayed the same.
        """
        ...


HESSIAN_REUSE_OPTIONS = (
    "reset",
    "keep",
    "keep_unless_penalty_increased",
    "keep_scaled",
)
"""The choices for `OptimistixStateAccess.hessian_reuse`."""


class OptimistixStateAccess(eqx.Module):
    """Reads the state of the optimistix gradient-based minimisers.

    `BFGS`, `DFP`, `LBFGS`, `GradientDescent` and `NonlinearCG` keep the aux
    output of the last accepted point in `aux`, and its gradient in
    `f_info.grad`. Other minimisers, like `OptaxMinimiser`, have neither, and
    both methods return `None` for them. Whether the fields exist is decided
    when the solve is traced, so this adds no work to the compiled loop.

    The quasi-Newton minimisers also keep a Hessian approximation, in
    `f_info.hessian` or `f_info.hessian_inv`, and for `LBFGS` in the history of
    `hessian_update_state`. `hessian_reuse` decides whether a new inner solve
    starts from it or from the identity.
    """

    hessian_reuse: str = eqx.field(default="reset", static=True)
    """Whether a new inner solve starts from the previous Hessian approximation.

    `"reset"` starts from the identity. `"keep"` always starts from the previous
    approximation. A multiplier update only changes the Hessian of the augmented
    Lagrangian through the constraint curvature, but a penalty increase adds a
    large term that the old approximation lacks. `"keep_unless_penalty_increased"`
    therefore resets after a penalty increase. `"keep_scaled"` instead multiplies
    the approximation by the penalty ratio, since the penalty term dominates the
    curvature once the penalty is large. It needs a dense approximation, as BFGS
    and DFP keep. For `LBFGS` it behaves like `"keep_unless_penalty_increased"`.
    """

    def __check_init__(self):
        if self.hessian_reuse not in HESSIAN_REUSE_OPTIONS:
            raise ValueError(
                f"`hessian_reuse` must be one of {HESSIAN_REUSE_OPTIONS}, "
                f"not {self.hessian_reuse!r}."
            )

    def accepted_evaluation(self, inner_state: PyTree) -> Evaluation | None:
        """Return the `aux` field of the state, if it holds an `Evaluation`."""
        aux = getattr(inner_state, "aux", None)
        return aux if isinstance(aux, Evaluation) else None

    def accepted_gradient(self, inner_state: PyTree) -> PyTree[Array] | None:
        """Return `f_info.grad` of the state, if it has one."""
        function_info = getattr(inner_state, "f_info", None)
        return getattr(function_info, "grad", None)

    def prefers_gradient_stopping(self, inner_state: PyTree) -> bool:
        """Return whether the state belongs to a minimiser with inexact steps.

        The `Newton` minimiser records the mode of its Hessian in the state, and
        only the matrix-free mode takes inexact steps.
        """
        prepared = getattr(inner_state, "prepared", None)
        return getattr(getattr(prepared, "value", None), "mode", None) == "matrix_free"

    def transfer_state(
        self,
        old_state: PyTree,
        new_state: PyTree,
        penalty_ratio: Scalar,
    ) -> PyTree:
        """Copy the Hessian approximation into `new_state`, if `hessian_reuse` says so.

        The copy survives the first step of the new inner solve. That step
        accepts the starting point without moving, so the quasi-Newton update
        sees no change in `y` and leaves the approximation as it is.
        """
        if self.hessian_reuse == "reset":
            return new_state
        function_info = getattr(new_state, "f_info", None)
        names = [
            name
            for name in ("hessian", "hessian_inv")
            if getattr(function_info, name, None) is not None
        ]
        if not names:
            return new_state
        name = names[0]
        operator = getattr(old_state.f_info, name)
        scalable = isinstance(operator, lx.PyTreeLinearOperator)
        if self.hessian_reuse == "keep_scaled" and scalable:
            # The Hessian grows with the penalty, and its inverse shrinks.
            factor = penalty_ratio if name == "hessian" else 1.0 / penalty_ratio
            operator = eqx.tree_at(
                lambda operator: operator.pytree,
                operator,
                jax.tree.map(lambda leaf: factor * leaf, operator.pytree),
            )
        copied = eqx.tree_at(
            lambda state: getattr(state.f_info, name), new_state, operator
        )
        if getattr(new_state, "hessian_update_state", None) is not None:
            copied = eqx.tree_at(
                lambda state: state.hessian_update_state,
                copied,
                old_state.hessian_update_state,
            )
        if self.hessian_reuse == "keep" or (
            self.hessian_reuse == "keep_scaled" and scalable
        ):
            return copied
        dynamic_copied, static = eqx.partition(copied, eqx.is_array)
        dynamic_new, _ = eqx.partition(new_state, eqx.is_array)
        dynamic = tree_where(penalty_ratio > 1.0, dynamic_new, dynamic_copied)
        return eqx.combine(dynamic, static)


class InnerStoppingRule(Protocol):
    """A rule that decides when an inner solve is finished."""

    def is_inner_converged(
        self,
        gradient: PyTree[Array] | None,
        tolerance: Scalar,
        gradient_scale: Scalar,
        minimiser_converged: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Return whether the inner solve has converged.

        `gradient` is the gradient of the augmented Lagrangian at the accepted
        point, or `None` if the inner solver does not provide it. `tolerance` is
        the current inner tolerance, and `gradient_scale` is the largest element
        of the objective gradient at `y0`, at least one. `minimiser_converged` is
        the inner minimiser's own verdict.
        """
        ...


class MinimiserStopping(eqx.Module):
    """Stops the inner solve when the inner minimiser says it has converged.

    The optimistix minimisers stop when `y` and the function value stop changing,
    measured against their `rtol` and `atol`.
    """

    def is_inner_converged(
        self,
        gradient: PyTree[Array] | None,
        tolerance: Scalar,
        gradient_scale: Scalar,
        minimiser_converged: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Return the inner minimiser's own verdict."""
        return minimiser_converged


class GradientStopping(eqx.Module):
    r"""Stops the inner solve when the gradient of $L_\rho$ is small.

    The inner solve stops once the largest element of the gradient is at most
    the inner tolerance, times the scale of the objective gradient at `y0` when
    `relative` is set. This is the criterion of ALGENCAN. For inner minimisers
    that do not provide the gradient, it uses the minimiser's own verdict.
    """

    relative: bool = eqx.field(default=True, static=True)
    """Whether to scale the tolerance with the objective gradient at `y0`."""

    def is_inner_converged(
        self,
        gradient: PyTree[Array] | None,
        tolerance: Scalar,
        gradient_scale: Scalar,
        minimiser_converged: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Compare the largest gradient element with the scaled tolerance."""
        if gradient is None:
            return minimiser_converged
        scale = gradient_scale if self.relative else jnp.ones_like(gradient_scale)
        return tree_max_norm(gradient) <= tolerance * scale
