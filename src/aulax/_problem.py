from typing import Generic, Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float, PyTree, Scalar


Y = TypeVar("Y")
Args = TypeVar("Args")
State = TypeVar("State")
Aux = TypeVar("Aux")

# `ProblemFunction` only takes `Y` and `Args` as inputs, so the type checker
# requires them to be contravariant. `State` and `Aux` appear inside the
# invariant `Evaluation`, so they stay invariant.
_Y_contra = TypeVar("_Y_contra", contravariant=True)
_Args_contra = TypeVar("_Args_contra", contravariant=True)

ConstraintTree = PyTree[Float[Array, "..."]] | None
"""A pytree of constraint residuals, or `None` when there are no constraints."""


class Evaluation(eqx.Module, Generic[State, Aux]):
    """The output of a problem function at one point.

    The equality residuals $h(y)$ are driven to zero, and the inequality residuals
    $g(y)$ are driven to non-positive values. Leave a field at `None` when the
    problem has no constraints of that kind.
    """

    objective: Scalar
    """Scalar objective value at the evaluated point."""
    equality: ConstraintTree = None
    """Equality residuals, which the solver drives to zero."""
    inequality: ConstraintTree = None
    """Inequality residuals, which the solver drives to non-positive values."""
    # A generic field cannot have a typed default, so `None` is the default for
    # problems that thread no state and return no aux output.
    state: State = None  # ty: ignore[invalid-assignment]
    """State passed to the next evaluation of the problem function.

    It must keep the same pytree structure, shapes and dtypes across evaluations,
    and the other outputs must not depend on it beyond the tolerance of whatever
    computation it warm-starts. The solver does not differentiate through it.
    """
    aux: Aux = None  # ty: ignore[invalid-assignment]
    """Auxiliary output, returned to the user and never read by the solver."""


class ProblemFunction(Protocol[_Y_contra, _Args_contra, State, Aux]):
    """A function that evaluates the objective and constraints of a problem.

    It is called as `problem(y, args, state)` and returns an `Evaluation`. Any
    function with this signature satisfies the protocol.
    """

    def __call__(
        self, y: _Y_contra, args: _Args_contra, state: State, /
    ) -> Evaluation[State, Aux]:
        """Evaluate the problem at `y`."""
        ...


class Constraints(eqx.Module):
    """A pair of pytrees shaped like the equality and inequality residuals.

    The solver uses it for anything that has one value per constraint, such as
    the multipliers, the penalties and the residual measures.
    """

    equality: ConstraintTree
    """Values for the equality constraints."""
    inequality: ConstraintTree
    """Values for the inequality constraints."""
