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


def _bound_residuals(y, lower, upper):
    """Return the residuals of the finite bounds `lower <= y <= upper`.

    Infinite entries mean no bound, and they get no residual at all, so the
    residual shapes are fixed when the problem is defined.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower_index = np.flatnonzero(np.isfinite(lower))
    upper_index = np.flatnonzero(np.isfinite(upper))
    return {
        "lower_bounds": lower[lower_index] - y[lower_index],
        "upper_bounds": y[upper_index] - upper[upper_index],
    }


INF = np.inf
"""An absent bound in `_bound_residuals`."""


def _rosenbrock(y):
    return 100.0 * (y[1] - y[0] ** 2) ** 2 + (1.0 - y[0]) ** 2


def hs1(y, args, state):
    return alx.Evaluation(
        objective=_rosenbrock(y),
        inequality=_bound_residuals(y, [-INF, -1.5], [INF, INF]),
    )


def hs2(y, args, state):
    return alx.Evaluation(
        objective=_rosenbrock(y),
        inequality=_bound_residuals(y, [-INF, 1.5], [INF, INF]),
    )


def hs3(y, args, state):
    return alx.Evaluation(
        objective=y[1] + 1e-5 * (y[1] - y[0]) ** 2,
        inequality=_bound_residuals(y, [-INF, 0.0], [INF, INF]),
    )


def hs4(y, args, state):
    return alx.Evaluation(
        objective=(y[0] + 1.0) ** 3 / 3.0 + y[1],
        inequality=_bound_residuals(y, [1.0, 0.0], [INF, INF]),
    )


def hs5(y, args, state):
    return alx.Evaluation(
        objective=jnp.sin(y[0] + y[1])
        + (y[0] - y[1]) ** 2
        - 1.5 * y[0]
        + 2.5 * y[1]
        + 1.0,
        inequality=_bound_residuals(y, [-1.5, -3.0], [4.0, 3.0]),
    )


def hs11(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 5.0) ** 2 + y[1] ** 2 - 25.0,
        inequality=y[0] ** 2 - y[1],
    )


def hs12(y, args, state):
    return alx.Evaluation(
        objective=0.5 * y[0] ** 2 + y[1] ** 2 - y[0] * y[1] - 7.0 * y[0] - 7.0 * y[1],
        inequality=4.0 * y[0] ** 2 + y[1] ** 2 - 25.0,
    )


def hs13(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 2.0) ** 2 + y[1] ** 2,
        inequality={
            "cubic": y[1] - (1.0 - y[0]) ** 3,
            **_bound_residuals(y, [0.0, 0.0], [INF, INF]),
        },
    )


def hs14(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 2.0) ** 2 + (y[1] - 1.0) ** 2,
        equality=y[0] - 2.0 * y[1] + 1.0,
        inequality=y[0] ** 2 / 4.0 + y[1] ** 2 - 1.0,
    )


def hs15(y, args, state):
    return alx.Evaluation(
        objective=_rosenbrock(y),
        inequality={
            "product": 1.0 - y[0] * y[1],
            "sum": -(y[0] + y[1] ** 2),
            **_bound_residuals(y, [-INF, -INF], [0.5, INF]),
        },
    )


def hs26(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2 + (y[1] - y[2]) ** 4,
        equality=(1.0 + y[1] ** 2) * y[0] + y[2] ** 4 - 3.0,
    )


def hs27(y, args, state):
    return alx.Evaluation(
        objective=0.01 * (y[0] - 1.0) ** 2 + (y[1] - y[0] ** 2) ** 2,
        equality=y[0] + y[2] ** 2 + 1.0,
    )


def hs29(y, args, state):
    return alx.Evaluation(
        objective=-y[0] * y[1] * y[2],
        inequality=y[0] ** 2 + 2.0 * y[1] ** 2 + 4.0 * y[2] ** 2 - 48.0,
    )


def hs30(y, args, state):
    return alx.Evaluation(
        objective=jnp.sum(y**2),
        inequality={
            "circle": 1.0 - y[0] ** 2 - y[1] ** 2,
            **_bound_residuals(y, [1.0, -10.0, -10.0], [10.0, 10.0, 10.0]),
        },
    )


def hs33(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 1.0) * (y[0] - 2.0) * (y[0] - 3.0) + y[2],
        inequality={
            "cone": y[0] ** 2 + y[1] ** 2 - y[2] ** 2,
            "sphere": 4.0 - jnp.sum(y**2),
            **_bound_residuals(y, [0.0, 0.0, 0.0], [INF, INF, 5.0]),
        },
    )


def hs34(y, args, state):
    return alx.Evaluation(
        objective=-y[0],
        inequality={
            "exponentials": jnp.array([jnp.exp(y[0]) - y[1], jnp.exp(y[1]) - y[2]]),
            **_bound_residuals(y, [0.0, 0.0, 0.0], [100.0, 100.0, 10.0]),
        },
    )


def hs38(y, args, state):
    objective = (
        100.0 * (y[1] - y[0] ** 2) ** 2
        + (1.0 - y[0]) ** 2
        + 90.0 * (y[3] - y[2] ** 2) ** 2
        + (1.0 - y[2]) ** 2
        + 10.1 * ((y[1] - 1.0) ** 2 + (y[3] - 1.0) ** 2)
        + 19.8 * (y[1] - 1.0) * (y[3] - 1.0)
    )
    return alx.Evaluation(
        objective=objective,
        inequality=_bound_residuals(y, [-10.0] * 4, [10.0] * 4),
    )


def hs39(y, args, state):
    return alx.Evaluation(
        objective=-y[0],
        equality=jnp.array(
            [y[1] - y[0] ** 3 - y[2] ** 2, y[0] ** 2 - y[1] - y[3] ** 2]
        ),
    )


def hs40(y, args, state):
    return alx.Evaluation(
        objective=-jnp.prod(y),
        equality=jnp.array(
            [
                y[0] ** 3 + y[1] ** 2 - 1.0,
                y[0] ** 2 * y[3] - y[2],
                y[3] ** 2 - y[1],
            ]
        ),
    )


def hs43(y, args, state):
    objective = (
        y[0] ** 2
        + y[1] ** 2
        + 2.0 * y[2] ** 2
        + y[3] ** 2
        - 5.0 * y[0]
        - 5.0 * y[1]
        - 21.0 * y[2]
        + 7.0 * y[3]
    )
    return alx.Evaluation(
        objective=objective,
        inequality=jnp.array(
            [
                jnp.sum(y**2) + y[0] - y[1] + y[2] - y[3] - 8.0,
                y[0] ** 2
                + 2.0 * y[1] ** 2
                + y[2] ** 2
                + 2.0 * y[3] ** 2
                - y[0]
                - y[3]
                - 10.0,
                2.0 * y[0] ** 2
                + y[1] ** 2
                + y[2] ** 2
                + 2.0 * y[0]
                - y[1]
                - y[3]
                - 5.0,
            ]
        ),
    )


def hs44(y, args, state):
    objective = (
        y[0] - y[1] - y[2] - y[0] * y[2] + y[0] * y[3] + y[1] * y[2] - y[1] * y[3]
    )
    return alx.Evaluation(
        objective=objective,
        inequality={
            "linear": jnp.array(
                [
                    y[0] + 2.0 * y[1] - 8.0,
                    4.0 * y[0] + y[1] - 12.0,
                    3.0 * y[0] + 4.0 * y[1] - 12.0,
                    2.0 * y[2] + y[3] - 8.0,
                    y[2] + 2.0 * y[3] - 8.0,
                    y[2] + y[3] - 5.0,
                ]
            ),
            **_bound_residuals(y, [0.0] * 4, [INF] * 4),
        },
    )


def hs46(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2
        + (y[2] - 1.0) ** 2
        + (y[3] - 1.0) ** 4
        + (y[4] - 1.0) ** 6,
        equality=jnp.array(
            [
                y[0] ** 2 * y[3] + jnp.sin(y[3] - y[4]) - 1.0,
                y[1] + y[2] ** 4 * y[3] ** 2 - 2.0,
            ]
        ),
    )


def hs47(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2
        + (y[1] - y[2]) ** 3
        + (y[2] - y[3]) ** 4
        + (y[3] - y[4]) ** 4,
        equality=jnp.array(
            [
                y[0] + y[1] ** 2 + y[2] ** 3 - 3.0,
                y[1] - y[2] ** 2 + y[3] - 1.0,
                y[0] * y[4] - 1.0,
            ]
        ),
    )


def hs48(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 1.0) ** 2 + (y[1] - y[2]) ** 2 + (y[3] - y[4]) ** 2,
        equality=jnp.array([jnp.sum(y) - 5.0, y[2] - 2.0 * (y[3] + y[4]) + 3.0]),
    )


def hs49(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2
        + (y[2] - 1.0) ** 2
        + (y[3] - 1.0) ** 4
        + (y[4] - 1.0) ** 6,
        equality=jnp.array(
            [y[0] + y[1] + y[2] + 4.0 * y[3] - 7.0, y[2] + 5.0 * y[4] - 6.0]
        ),
    )


def hs50(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2
        + (y[1] - y[2]) ** 2
        + (y[2] - y[3]) ** 4
        + (y[3] - y[4]) ** 2,
        equality=jnp.array(
            [
                y[0] + 2.0 * y[1] + 3.0 * y[2] - 6.0,
                y[1] + 2.0 * y[2] + 3.0 * y[3] - 6.0,
                y[2] + 2.0 * y[3] + 3.0 * y[4] - 6.0,
            ]
        ),
    )


def _hs51_constraints(y, right_hand_side):
    return jnp.array(
        [
            y[0] + 3.0 * y[1] - right_hand_side,
            y[2] + y[3] - 2.0 * y[4],
            y[1] - y[4],
        ]
    )


def hs51(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2
        + (y[1] + y[2] - 2.0) ** 2
        + (y[3] - 1.0) ** 2
        + (y[4] - 1.0) ** 2,
        equality=_hs51_constraints(y, 4.0),
    )


def hs52(y, args, state):
    return alx.Evaluation(
        objective=(4.0 * y[0] - y[1]) ** 2
        + (y[1] + y[2] - 2.0) ** 2
        + (y[3] - 1.0) ** 2
        + (y[4] - 1.0) ** 2,
        equality=_hs51_constraints(y, 0.0),
    )


def hs60(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 1.0) ** 2 + (y[0] - y[1]) ** 2 + (y[1] - y[2]) ** 4,
        equality=y[0] * (1.0 + y[1] ** 2) + y[2] ** 4 - 4.0 - 3.0 * jnp.sqrt(2.0),
        inequality=_bound_residuals(y, [-10.0] * 3, [10.0] * 3),
    )


def hs63(y, args, state):
    return alx.Evaluation(
        objective=1000.0
        - y[0] ** 2
        - 2.0 * y[1] ** 2
        - y[2] ** 2
        - y[0] * y[1]
        - y[0] * y[2],
        equality=jnp.array(
            [
                8.0 * y[0] + 14.0 * y[1] + 7.0 * y[2] - 56.0,
                jnp.sum(y**2) - 25.0,
            ]
        ),
        inequality=_bound_residuals(y, [0.0] * 3, [INF] * 3),
    )


def hs64(y, args, state):
    return alx.Evaluation(
        objective=5.0 * y[0]
        + 50000.0 / y[0]
        + 20.0 * y[1]
        + 72000.0 / y[1]
        + 10.0 * y[2]
        + 144000.0 / y[2],
        inequality={
            "reciprocals": 4.0 / y[0] + 32.0 / y[1] + 120.0 / y[2] - 1.0,
            **_bound_residuals(y, [1e-5] * 3, [INF] * 3),
        },
    )


def hs65(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - y[1]) ** 2
        + (y[0] + y[1] - 10.0) ** 2 / 9.0
        + (y[2] - 5.0) ** 2,
        inequality={
            "sphere": jnp.sum(y**2) - 48.0,
            **_bound_residuals(y, [-4.5, -4.5, -5.0], [4.5, 4.5, 5.0]),
        },
    )


def hs66(y, args, state):
    return alx.Evaluation(
        objective=0.2 * y[2] - 0.8 * y[0],
        inequality={
            "exponentials": jnp.array([jnp.exp(y[0]) - y[1], jnp.exp(y[1]) - y[2]]),
            **_bound_residuals(y, [0.0, 0.0, 0.0], [100.0, 100.0, 10.0]),
        },
    )


def hs72(y, args, state):
    return alx.Evaluation(
        objective=1.0 + jnp.sum(y),
        inequality={
            "reciprocals": jnp.array(
                [
                    4.0 / y[0] + 2.25 / y[1] + 1.0 / y[2] + 0.25 / y[3] - 0.0401,
                    0.16 / y[0] + 0.36 / y[1] + 0.64 / y[2] + 0.64 / y[3] - 0.010085,
                ]
            ),
            **_bound_residuals(y, [0.001] * 4, [4e5, 3e5, 2e5, 1e5]),
        },
    )


def _hs74_75(y, bound):
    return alx.Evaluation(
        objective=3.0 * y[0] + 1e-6 * y[0] ** 3 + 2.0 * y[1] + 2e-6 / 3.0 * y[1] ** 3,
        equality=jnp.array(
            [
                1000.0 * jnp.sin(-y[2] - 0.25)
                + 1000.0 * jnp.sin(-y[3] - 0.25)
                + 894.8
                - y[0],
                1000.0 * jnp.sin(y[2] - 0.25)
                + 1000.0 * jnp.sin(y[2] - y[3] - 0.25)
                + 894.8
                - y[1],
                1000.0 * jnp.sin(y[3] - 0.25)
                + 1000.0 * jnp.sin(y[3] - y[2] - 0.25)
                + 1294.8,
            ]
        ),
        inequality={
            "angles": jnp.array([y[2] - y[3] - bound, y[3] - y[2] - bound]),
            **_bound_residuals(
                y, [0.0, 0.0, -bound, -bound], [1200.0, 1200.0, bound, bound]
            ),
        },
    )


def hs74(y, args, state):
    return _hs74_75(y, 0.55)


def hs75(y, args, state):
    return _hs74_75(y, 0.48)


def hs76(y, args, state):
    objective = (
        y[0] ** 2
        + 0.5 * y[1] ** 2
        + y[2] ** 2
        + 0.5 * y[3] ** 2
        - y[0] * y[2]
        + y[2] * y[3]
        - y[0]
        - 3.0 * y[1]
        + y[2]
        - y[3]
    )
    return alx.Evaluation(
        objective=objective,
        inequality={
            "linear": jnp.array(
                [
                    y[0] + 2.0 * y[1] + y[2] + y[3] - 5.0,
                    3.0 * y[0] + y[1] + 2.0 * y[2] - y[3] - 4.0,
                    1.5 - y[1] - 4.0 * y[2],
                ]
            ),
            **_bound_residuals(y, [0.0] * 4, [INF] * 4),
        },
    )


def hs77(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 1.0) ** 2
        + (y[0] - y[1]) ** 2
        + (y[2] - 1.0) ** 2
        + (y[3] - 1.0) ** 4
        + (y[4] - 1.0) ** 6,
        equality=jnp.array(
            [
                y[0] ** 2 * y[3] + jnp.sin(y[3] - y[4]) - 2.0 * jnp.sqrt(2.0),
                y[1] + y[2] ** 4 * y[3] ** 2 - 8.0 - jnp.sqrt(2.0),
            ]
        ),
    )


def _hs78_constraints(y):
    return jnp.array(
        [
            jnp.sum(y**2) - 10.0,
            y[1] * y[2] - 5.0 * y[3] * y[4],
            y[0] ** 3 + y[1] ** 3 + 1.0,
        ]
    )


def hs78(y, args, state):
    return alx.Evaluation(objective=jnp.prod(y), equality=_hs78_constraints(y))


def hs79(y, args, state):
    return alx.Evaluation(
        objective=(y[0] - 1.0) ** 2
        + (y[0] - y[1]) ** 2
        + (y[1] - y[2]) ** 2
        + (y[2] - y[3]) ** 4
        + (y[3] - y[4]) ** 4,
        equality=jnp.array(
            [
                y[0] + y[1] ** 2 + y[2] ** 3 - 2.0 - 3.0 * jnp.sqrt(2.0),
                y[1] - y[2] ** 2 + y[3] + 2.0 - 2.0 * jnp.sqrt(2.0),
                y[0] * y[4] - 2.0,
            ]
        ),
    )


_HS80_LOWER = [-2.3, -2.3, -3.2, -3.2, -3.2]
_HS80_UPPER = [2.3, 2.3, 3.2, 3.2, 3.2]


def hs80(y, args, state):
    return alx.Evaluation(
        objective=jnp.exp(jnp.prod(y)),
        equality=_hs78_constraints(y),
        inequality=_bound_residuals(y, _HS80_LOWER, _HS80_UPPER),
    )


def hs81(y, args, state):
    return alx.Evaluation(
        objective=jnp.exp(jnp.prod(y)) - 0.5 * (y[0] ** 3 + y[1] ** 3 + 1.0) ** 2,
        equality=_hs78_constraints(y),
        inequality=_bound_residuals(y, _HS80_LOWER, _HS80_UPPER),
    )


def hs100(y, args, state):
    objective = (
        (y[0] - 10.0) ** 2
        + 5.0 * (y[1] - 12.0) ** 2
        + y[2] ** 4
        + 3.0 * (y[3] - 11.0) ** 2
        + 10.0 * y[4] ** 6
        + 7.0 * y[5] ** 2
        + y[6] ** 4
        - 4.0 * y[5] * y[6]
        - 10.0 * y[5]
        - 8.0 * y[6]
    )
    return alx.Evaluation(
        objective=objective,
        inequality=jnp.array(
            [
                2.0 * y[0] ** 2
                + 3.0 * y[1] ** 4
                + y[2]
                + 4.0 * y[3] ** 2
                + 5.0 * y[4]
                - 127.0,
                7.0 * y[0] + 3.0 * y[1] + 10.0 * y[2] ** 2 + y[3] - y[4] - 282.0,
                23.0 * y[0] + y[1] ** 2 + 6.0 * y[5] ** 2 - 8.0 * y[6] - 196.0,
                4.0 * y[0] ** 2
                + y[1] ** 2
                - 3.0 * y[0] * y[1]
                + 2.0 * y[2] ** 2
                + 5.0 * y[5]
                - 11.0 * y[6],
            ]
        ),
    )


def hs113(y, args, state):
    objective = (
        y[0] ** 2
        + y[1] ** 2
        + y[0] * y[1]
        - 14.0 * y[0]
        - 16.0 * y[1]
        + (y[2] - 10.0) ** 2
        + 4.0 * (y[3] - 5.0) ** 2
        + (y[4] - 3.0) ** 2
        + 2.0 * (y[5] - 1.0) ** 2
        + 5.0 * y[6] ** 2
        + 7.0 * (y[7] - 11.0) ** 2
        + 2.0 * (y[8] - 10.0) ** 2
        + (y[9] - 7.0) ** 2
        + 45.0
    )
    return alx.Evaluation(
        objective=objective,
        inequality=-jnp.array(
            [
                105.0 - 4.0 * y[0] - 5.0 * y[1] + 3.0 * y[6] - 9.0 * y[7],
                -10.0 * y[0] + 8.0 * y[1] + 17.0 * y[6] - 2.0 * y[7],
                8.0 * y[0] - 2.0 * y[1] - 5.0 * y[8] + 2.0 * y[9] + 12.0,
                -3.0 * (y[0] - 2.0) ** 2
                - 4.0 * (y[1] - 3.0) ** 2
                - 2.0 * y[2] ** 2
                + 7.0 * y[3]
                + 120.0,
                -5.0 * y[0] ** 2 - 8.0 * y[1] - (y[2] - 6.0) ** 2 + 2.0 * y[3] + 40.0,
                -0.5 * (y[0] - 8.0) ** 2
                - 2.0 * (y[1] - 4.0) ** 2
                - 3.0 * y[4] ** 2
                + y[5]
                + 30.0,
                -(y[0] ** 2)
                - 2.0 * (y[1] - 2.0) ** 2
                + 2.0 * y[0] * y[1]
                - 14.0 * y[4]
                + 6.0 * y[5],
                3.0 * y[0] - 6.0 * y[1] - 12.0 * (y[8] - 8.0) ** 2 + 7.0 * y[9],
            ]
        ),
    )


PROBLEMS = [
    BenchmarkProblem(
        name="hs1",
        problem=hs1,
        y0=np.array([-2.0, 1.0]),
        solution=np.array([1.0, 1.0]),
        objective=0.0,
        tags=frozenset({"bounds"}),
    ),
    BenchmarkProblem(
        name="hs2",
        problem=hs2,
        y0=np.array([-2.0, 1.0]),
        solution=np.array([1.224370749, 1.5]),
        objective=0.0504261879,
        tags=frozenset({"bounds"}),
    ),
    BenchmarkProblem(
        name="hs3",
        problem=hs3,
        y0=np.array([10.0, 1.0]),
        solution=np.array([0.0, 0.0]),
        objective=0.0,
        tags=frozenset({"bounds"}),
    ),
    BenchmarkProblem(
        name="hs4",
        problem=hs4,
        y0=np.array([1.125, 0.125]),
        solution=np.array([1.0, 0.0]),
        objective=8.0 / 3.0,
        tags=frozenset({"bounds"}),
    ),
    BenchmarkProblem(
        name="hs5",
        problem=hs5,
        y0=np.array([0.0, 0.0]),
        solution=np.array([0.5 - np.pi / 3.0, -0.5 - np.pi / 3.0]),
        objective=-np.sqrt(3.0) / 2.0 - np.pi / 3.0,
        tags=frozenset({"bounds"}),
    ),
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
        name="hs11",
        problem=hs11,
        y0=np.array([4.9, 0.1]),
        objective=-8.498464223,
        tags=frozenset({"inequality"}),
    ),
    BenchmarkProblem(
        name="hs12",
        problem=hs12,
        y0=np.array([0.0, 0.0]),
        solution=np.array([2.0, 3.0]),
        objective=-30.0,
        tags=frozenset({"inequality"}),
    ),
    BenchmarkProblem(
        name="hs13",
        problem=hs13,
        y0=np.array([-2.0, -2.0]),
        solution=np.array([1.0, 0.0]),
        objective=1.0,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs14",
        problem=hs14,
        y0=np.array([2.0, 2.0]),
        solution=np.array([(np.sqrt(7.0) - 1.0) / 2.0, (np.sqrt(7.0) + 1.0) / 4.0]),
        objective=9.0 - 23.0 * np.sqrt(7.0) / 8.0,
        tags=frozenset({"equality", "inequality"}),
    ),
    BenchmarkProblem(
        name="hs15",
        problem=hs15,
        y0=np.array([-2.0, 1.0]),
        solution=np.array([0.5, 2.0]),
        objective=306.5,
        tags=frozenset({"inequality", "bounds"}),
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
        name="hs26",
        problem=hs26,
        y0=np.array([-2.6, 2.0, 2.0]),
        solution=np.array([1.0, 1.0, 1.0]),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs27",
        problem=hs27,
        y0=np.array([2.0, 2.0, 2.0]),
        solution=np.array([-1.0, 1.0, 0.0]),
        objective=0.04,
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
        name="hs29",
        problem=hs29,
        y0=np.array([1.0, 1.0, 1.0]),
        objective=-16.0 * np.sqrt(2.0),
        tags=frozenset({"inequality"}),
    ),
    BenchmarkProblem(
        name="hs30",
        problem=hs30,
        y0=np.array([1.0, 1.0, 1.0]),
        solution=np.array([1.0, 0.0, 0.0]),
        objective=1.0,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs33",
        problem=hs33,
        y0=np.array([0.0, 0.0, 3.0]),
        solution=np.array([0.0, np.sqrt(2.0), np.sqrt(2.0)]),
        objective=np.sqrt(2.0) - 6.0,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs34",
        problem=hs34,
        y0=np.array([0.0, 1.05, 2.9]),
        solution=np.array([np.log(np.log(10.0)), np.log(10.0), 10.0]),
        objective=-np.log(np.log(10.0)),
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
        name="hs38",
        problem=hs38,
        y0=np.array([-3.0, -1.0, -3.0, -1.0]),
        solution=np.array([1.0, 1.0, 1.0, 1.0]),
        objective=0.0,
        tags=frozenset({"bounds"}),
    ),
    BenchmarkProblem(
        name="hs39",
        problem=hs39,
        y0=np.array([2.0, 2.0, 2.0, 2.0]),
        solution=np.array([1.0, 1.0, 0.0, 0.0]),
        objective=-1.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs40",
        problem=hs40,
        y0=np.array([0.8, 0.8, 0.8, 0.8]),
        solution=np.array(
            [2.0 ** (-1 / 3), 2.0 ** (-1 / 2), 2.0 ** (-11 / 12), 2.0 ** (-1 / 4)]
        ),
        objective=-0.25,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs43",
        problem=hs43,
        y0=np.array([0.0, 0.0, 0.0, 0.0]),
        solution=np.array([0.0, 1.0, 2.0, -1.0]),
        objective=-44.0,
        tags=frozenset({"inequality"}),
    ),
    BenchmarkProblem(
        name="hs44",
        problem=hs44,
        y0=np.array([0.0, 0.0, 0.0, 0.0]),
        solution=np.array([0.0, 3.0, 0.0, 4.0]),
        objective=-15.0,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs46",
        problem=hs46,
        y0=np.array([np.sqrt(2.0) / 2.0, 1.75, 0.5, 2.0, 2.0]),
        solution=np.array([1.0] * 5),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs47",
        problem=hs47,
        y0=np.array([2.0, np.sqrt(2.0), -1.0, 2.0 - np.sqrt(2.0), 0.5]),
        solution=np.array([1.0] * 5),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs48",
        problem=hs48,
        y0=np.array([3.0, 5.0, -3.0, 2.0, -2.0]),
        solution=np.array([1.0] * 5),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs49",
        problem=hs49,
        y0=np.array([10.0, 7.0, 2.0, -3.0, 0.8]),
        solution=np.array([1.0] * 5),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs50",
        problem=hs50,
        y0=np.array([35.0, -31.0, 11.0, 5.0, -5.0]),
        solution=np.array([1.0] * 5),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs51",
        problem=hs51,
        y0=np.array([2.5, 0.5, 2.0, -1.0, 0.5]),
        solution=np.array([1.0] * 5),
        objective=0.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs52",
        problem=hs52,
        y0=np.array([2.0] * 5),
        solution=np.array(np.array([-33.0, 11.0, 180.0, -158.0, 11.0]) / 349.0),
        objective=1859.0 / 349.0,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs60",
        problem=hs60,
        y0=np.array([2.0, 2.0, 2.0]),
        solution=np.array([1.104859024, 1.196674194, 1.535262257]),
        objective=0.0325682003,
        tags=frozenset({"equality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs63",
        problem=hs63,
        y0=np.array([2.0, 2.0, 2.0]),
        solution=np.array([3.512118414, 0.2169881741, 3.552174034]),
        objective=961.7151721,
        tags=frozenset({"equality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs64",
        problem=hs64,
        y0=np.array([1.0, 1.0, 1.0]),
        solution=np.array([108.7347175, 85.12613942, 204.3247078]),
        objective=6299.842428,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs65",
        problem=hs65,
        y0=np.array([-5.0, 5.0, 0.0]),
        solution=np.array([3.650461821, 3.65046168, 4.6204170507]),
        objective=0.9535288567,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs66",
        problem=hs66,
        y0=np.array([0.0, 1.05, 2.9]),
        solution=np.array([0.1841264879, 1.202167873, 3.327322322]),
        objective=0.5181632741,
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
    BenchmarkProblem(
        name="hs72",
        problem=hs72,
        y0=np.array([1.0, 1.0, 1.0, 1.0]),
        solution=np.array([193.4071, 179.5475, 185.0186, 168.7062]),
        objective=727.67937,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs74",
        problem=hs74,
        y0=np.array([0.0, 0.0, 0.0, 0.0]),
        solution=np.array([679.9453, 1026.067, 0.1188764, -0.3962336]),
        objective=5126.4981,
        tags=frozenset({"equality", "inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs75",
        problem=hs75,
        y0=np.array([0.0, 0.0, 0.0, 0.0]),
        solution=np.array([776.1592, 925.1949, 0.05110879, -0.4288911]),
        objective=5174.4129,
        tags=frozenset({"equality", "inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs76",
        problem=hs76,
        y0=np.array([0.5, 0.5, 0.5, 0.5]),
        solution=np.array([0.2727273, 2.090909, 0.0, 0.5454545]),
        objective=-4.681818181,
        tags=frozenset({"inequality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs77",
        problem=hs77,
        y0=np.array([2.0] * 5),
        solution=np.array([1.166172, 1.182111, 1.380257, 1.506036, 0.6109203]),
        objective=0.24150513,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs78",
        problem=hs78,
        y0=np.array([-2.0, 1.5, 2.0, -1.0, -1.0]),
        solution=np.array([-1.717142, 1.595708, 1.827248, -0.7636429, -0.7636435]),
        objective=-2.919700,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs79",
        problem=hs79,
        y0=np.array([2.0] * 5),
        solution=np.array([1.191127, 1.362603, 1.472818, 1.635017, 1.679081]),
        objective=0.0787768209,
        tags=frozenset({"equality"}),
    ),
    BenchmarkProblem(
        name="hs80",
        problem=hs80,
        y0=np.array([-2.0, 2.0, 2.0, -1.0, -1.0]),
        solution=np.array([-1.717143, 1.595709, 1.827247, -0.7636413, -0.7636450]),
        objective=0.0539498478,
        tags=frozenset({"equality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs81",
        problem=hs81,
        y0=np.array([-2.0, 2.0, 2.0, -1.0, -1.0]),
        solution=np.array([-1.717143, 1.595709, 1.827247, -0.7636413, -0.7636450]),
        objective=0.0539498478,
        tags=frozenset({"equality", "bounds"}),
    ),
    BenchmarkProblem(
        name="hs100",
        problem=hs100,
        y0=np.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0]),
        solution=np.array(
            [2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227]
        ),
        objective=680.6300573,
        tags=frozenset({"inequality"}),
    ),
    BenchmarkProblem(
        name="hs113",
        problem=hs113,
        y0=np.array([2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 7.0, 3.0, 6.0, 10.0]),
        solution=np.array(
            [
                2.171996,
                2.363683,
                8.773926,
                5.095984,
                0.9906548,
                1.430574,
                1.321644,
                9.828726,
                8.280092,
                8.375927,
            ]
        ),
        objective=24.3062091,
        tags=frozenset({"inequality"}),
    ),
]
"""The problems, each with the starting point and solution from the book."""
