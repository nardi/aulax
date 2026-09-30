from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Scalar

from ._misc import tree_max_norm
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


class OptimistixStateAccess(eqx.Module):
    """Reads the state of the optimistix gradient-based minimisers.

    `BFGS`, `DFP`, `LBFGS`, `GradientDescent` and `NonlinearCG` keep the aux
    output of the last accepted point in `aux`, and its gradient in
    `f_info.grad`. Other minimisers, like `OptaxMinimiser`, have neither, and
    both methods return `None` for them. Whether the fields exist is decided
    when the solve is traced, so this adds no work to the compiled loop.
    """

    def accepted_evaluation(self, inner_state: PyTree) -> Evaluation | None:
        """Return the `aux` field of the state, if it holds an `Evaluation`."""
        aux = getattr(inner_state, "aux", None)
        return aux if isinstance(aux, Evaluation) else None

    def accepted_gradient(self, inner_state: PyTree) -> PyTree[Array] | None:
        """Return `f_info.grad` of the state, if it has one."""
        function_info = getattr(inner_state, "f_info", None)
        return getattr(function_info, "grad", None)


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
