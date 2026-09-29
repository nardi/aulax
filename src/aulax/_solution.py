from typing import Generic

import equinox as eqx
import optimistix as optx
from jaxtyping import Array, Int, Scalar

from ._problem import Aux, Constraints, State, Y


# Equinox enumerations gain new items through subclassing, the same way
# `optimistix.RESULTS` extends `lineax.RESULTS`. ty reads the enumeration as a
# final `enum.Enum` and its members as strings, so it needs an ignore here and
# wherever a member is passed as a `RESULTS`.
class RESULTS(optx.RESULTS):  # ty: ignore[subclass-of-final-class]
    """The outcome of a solve, extending `optimistix.RESULTS` with `infeasible`."""

    infeasible = (
        "The penalty reached its maximum while the constraint violation stopped "
        "decreasing. The problem may be infeasible, or its constraints may be badly "
        "scaled."
    )


class Statistics(eqx.Module):
    """Statistics about a solve."""

    num_steps: Int[Array, ""]
    """The number of inner solver steps, summed over all inner solves."""
    num_outer_steps: Int[Array, ""]
    """The number of multiplier updates."""
    penalty: Constraints
    """The penalty at the end of the solve, one value per constraint element."""
    infeasibility: Scalar
    r"""The largest violation of feasibility or complementarity at the solution.

    This is the larger of $\lVert h \rVert_\infty$ and
    $\lVert \min(-g, \mu) \rVert_\infty$.
    """
    stationarity: Scalar
    r"""The largest element of the gradient of the Lagrangian at the solution."""


class Solution(eqx.Module, Generic[Y, State, Aux]):
    """The solution of a constrained minimization."""

    value: Y
    """The final value of `y`."""
    result: RESULTS
    """Whether the solve succeeded. `alx.RESULTS[result]` returns the message."""
    multipliers: Constraints
    """The Lagrange multipliers at the solution, shaped like the constraints."""
    state: State
    """The problem state from the last evaluation at the solution.

    It can warm-start a later solve of a nearby problem, together with `value`
    and `multipliers`.
    """
    aux: Aux
    """The aux output of the problem at the solution."""
    stats: Statistics
    """Statistics about the solve."""
