"""Hock-Schittkowski test problems with known solutions.

The numbering follows W. Hock and K. Schittkowski, Test Examples for Nonlinear
Programming Codes, 1981. Inequality constraints are written as g(y) <= 0, so the
book's g(y) >= 0 constraints appear here with their sign flipped.
"""

import jax.numpy as jnp
import numpy as np

import aulax as alx

from ._types import BenchmarkProblem


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
    BenchmarkProblem(
        name="hs6",
        problem=hs6,
        y0=np.array([-1.2, 1.0]),
        solution=np.array([1.0, 1.0]),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs7",
        problem=hs7,
        y0=np.array([2.0, 2.0]),
        solution=np.array([0.0, np.sqrt(3.0)]),
        objective=-np.sqrt(3.0),
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs28",
        problem=hs28,
        y0=np.array([-4.0, 1.0, 1.0]),
        solution=np.array([0.5, -0.5, 0.5]),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs21",
        problem=hs21,
        y0=np.array([-1.0, -1.0]),
        solution=np.array([2.0, 0.0]),
        objective=-99.96,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs35",
        problem=hs35,
        y0=np.array([0.5, 0.5, 0.5]),
        solution=np.array([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0]),
        objective=1.0 / 9.0,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs71",
        problem=hs71,
        y0=np.array([1.0, 5.0, 5.0, 1.0]),
        solution=np.array([1.0, 4.7429994, 3.8211503, 1.3794082]),
        objective=17.0140173,
        tags=frozenset({"equality", "inequality", "bounds"}),
    ),
]
"""The problems, each with the starting point and solution from the book."""
