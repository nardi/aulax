"""Hock-Schittkowski test problems with known solutions.

The numbering follows W. Hock and K. Schittkowski, Test Examples for Nonlinear
Programming Codes, 1981. Inequality constraints are written as g(y) <= 0, so the
book's g(y) >= 0 constraints appear here with their sign flipped.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

import aulax as alx


@dataclass(frozen=True)
class KnownProblem:
    """A problem function with its starting point and known solution."""

    name: str
    """The name of the problem in the book."""
    problem: alx.ProblemFunction[Float[Array, " n"], None, None, None]
    """The problem function."""
    y0: np.ndarray
    """The starting point from the book."""
    solution: np.ndarray
    """The known minimizer."""
    objective: float
    """The objective value at the minimizer."""


def hs6(y, args, state):
    return alx.Evaluation(
        objective=(1.0 - y[0]) ** 2,
        equality=10.0 * (y[1] - y[0] ** 2),
    )


def hs7(y, args, state):
    return alx.Evaluation(
        objective=jnp.log(1.0 + y[0] ** 2) - y[1],
        equality=(1.0 + y[0] ** 2) ** 2 + y[1] ** 2 - 4.0,
    )


def hs28(y, args, state):
    return alx.Evaluation(
        objective=(y[0] + y[1]) ** 2 + (y[1] + y[2]) ** 2,
        equality=y[0] + 2.0 * y[1] + 3.0 * y[2] - 1.0,
    )


def hs21(y, args, state):
    return alx.Evaluation(
        objective=0.01 * y[0] ** 2 + y[1] ** 2 - 100.0,
        inequality=jnp.array(
            [
                10.0 - 10.0 * y[0] + y[1],
                2.0 - y[0],
                y[0] - 50.0,
                -50.0 - y[1],
                y[1] - 50.0,
            ]
        ),
    )


def hs35(y, args, state):
    objective = (
        9.0
        - 8.0 * y[0]
        - 6.0 * y[1]
        - 4.0 * y[2]
        + 2.0 * y[0] ** 2
        + 2.0 * y[1] ** 2
        + y[2] ** 2
        + 2.0 * y[0] * y[1]
        + 2.0 * y[0] * y[2]
    )
    return alx.Evaluation(
        objective=objective,
        inequality={
            "sum": y[0] + y[1] + 2.0 * y[2] - 3.0,
            "bounds": -y,
        },
    )


def hs71(y, args, state):
    return alx.Evaluation(
        objective=y[0] * y[3] * (y[0] + y[1] + y[2]) + y[2],
        equality=jnp.sum(y**2) - 40.0,
        inequality=jnp.concatenate([jnp.array([25.0 - jnp.prod(y)]), 1.0 - y, y - 5.0]),
    )


PROBLEMS = [
    KnownProblem("hs6", hs6, np.array([-1.2, 1.0]), np.array([1.0, 1.0]), 0.0),
    KnownProblem(
        "hs7", hs7, np.array([2.0, 2.0]), np.array([0.0, np.sqrt(3.0)]), -np.sqrt(3.0)
    ),
    KnownProblem(
        "hs28",
        hs28,
        np.array([-4.0, 1.0, 1.0]),
        np.array([0.5, -0.5, 0.5]),
        0.0,
    ),
    KnownProblem("hs21", hs21, np.array([-1.0, -1.0]), np.array([2.0, 0.0]), -99.96),
    KnownProblem(
        "hs35",
        hs35,
        np.array([0.5, 0.5, 0.5]),
        np.array([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0]),
        1.0 / 9.0,
    ),
    KnownProblem(
        "hs71",
        hs71,
        np.array([1.0, 5.0, 5.0, 1.0]),
        np.array([1.0, 4.7429994, 3.8211503, 1.3794082]),
        17.0140173,
    ),
]
