"""Problem families with a size parameter, to show how cost grows with size.

Each family is sparse, with no constraint that involves all variables, so the
work per evaluation grows linearly with the size.
"""

import jax.numpy as jnp
import numpy as np

import aulax as alx

from ._types import BenchmarkProblem


SIZES = (10, 100, 1000)
"""The sizes each family runs at."""


def simplex_projection(y, args, state):
    """Project `args` onto the probability simplex."""
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2),
        equality=jnp.sum(y) - 1.0,
        inequality=-y,
    )


def projected_onto_simplex(point: np.ndarray) -> np.ndarray:
    """Return the projection of `point` onto the probability simplex.

    This is the sorting algorithm of Held, Wolfe and Crowder: find the largest
    shift that keeps the positive entries summing to one, then clip at zero.
    """
    descending = np.sort(point)[::-1]
    cumulative = np.cumsum(descending) - 1.0
    indices = np.arange(1, point.size + 1)
    count = indices[descending - cumulative / indices > 0][-1]
    shift = cumulative[count - 1] / count
    return np.maximum(point - shift, 0.0)


def _simplex_problem(size: int) -> BenchmarkProblem:
    """Return the simplex projection of a fixed point with `size` entries."""
    point = np.linspace(-1.0, 1.0, size)
    solution = projected_onto_simplex(point)
    return BenchmarkProblem(
        name=f"simplex_{size}",
        problem=simplex_projection,
        y0=np.full(size, 1.0 / size),
        args=point,
        solution=solution,
        objective=float(np.sum((solution - point) ** 2)),
        tags=frozenset({"equality", "bounds"}),
    )


PENDULUM_HORIZON = 5.0
"""The time in which the pendulum has to swing up, in seconds."""

PENDULUM_MAXIMUM_TORQUE = 3.0
"""The largest control torque, too small to lift the pendulum in one go."""

PENDULUM_GRAVITY = 9.81
"""The gravitational acceleration divided by the pendulum length."""


def pendulum_dynamics(theta, omega, torque, step):
    """Return the residuals of the explicit Euler steps of a pendulum."""
    return {
        "theta": theta[1:] - theta[:-1] - step * omega[:-1],
        "omega": omega[1:]
        - omega[:-1]
        - step * (-PENDULUM_GRAVITY * jnp.sin(theta[:-1]) + torque),
    }


def pendulum_swing_up(y, args, state):
    """Swing a pendulum from hanging down to upright with the least control effort.

    `y` holds the angles `theta`, the angular velocities `omega`, and the
    torques `torque` on a grid of `N` steps. The dynamics are discretized with
    the explicit Euler method, and are equality constraints between neighbouring
    steps.
    """
    theta, omega, torque = y["theta"], y["omega"], y["torque"]
    step = PENDULUM_HORIZON / torque.size
    dynamics = pendulum_dynamics(theta, omega, torque, step)
    boundary = jnp.array([theta[0], omega[0], theta[-1] - jnp.pi, omega[-1]])
    return alx.Evaluation(
        objective=step * jnp.sum(torque**2),
        equality={"dynamics": dynamics, "boundary": boundary},
        inequality={
            "lower": -PENDULUM_MAXIMUM_TORQUE - torque,
            "upper": torque - PENDULUM_MAXIMUM_TORQUE,
        },
    )


def _pendulum_problem(steps: int) -> BenchmarkProblem:
    """Return the swing-up problem on a grid of `steps` time steps."""
    theta = np.linspace(0.0, np.pi, steps + 1)
    return BenchmarkProblem(
        name=f"pendulum_{steps}",
        problem=pendulum_swing_up,
        y0={
            "theta": theta,
            "omega": np.full(steps + 1, np.pi / PENDULUM_HORIZON),
            "torque": np.zeros(steps),
        },
        tags=frozenset({"equality", "bounds"}),
    )


POISSON_REGULARIZATION = 1e-3
r"""The weight $\alpha$ of the control cost in the Poisson problem."""

POISSON_MAXIMUM_CONTROL = 5.0
"""The bound on the control, below the $\\pi^2$ the unbounded solution needs."""


def poisson_control(y, args, state):
    r"""Steer the solution of $-u'' = z$ on $(0, 1)$ towards $\sin(\pi x)$.

    Both the state $u$ and the control $z$ are variables, on a grid of interior
    points with $u = 0$ at both ends. The discretized equation gives one equality
    constraint per grid point, and the control has bounds.
    """
    state_values, control = y["state"], y["control"]
    size = state_values.size
    step = 1.0 / (size + 1)
    grid = step * jnp.arange(1, size + 1)
    padded = jnp.pad(state_values, 1)
    laplacian = (2.0 * padded[1:-1] - padded[:-2] - padded[2:]) / step**2
    target = jnp.sin(jnp.pi * grid)
    return alx.Evaluation(
        objective=0.5 * step * jnp.sum((state_values - target) ** 2)
        + 0.5 * POISSON_REGULARIZATION * step * jnp.sum(control**2),
        equality=laplacian - control,
        inequality={
            "lower": -POISSON_MAXIMUM_CONTROL - control,
            "upper": control - POISSON_MAXIMUM_CONTROL,
        },
    )


def _poisson_problem(size: int) -> BenchmarkProblem:
    """Return the Poisson control problem on `size` interior grid points."""
    return BenchmarkProblem(
        name=f"poisson_{size}",
        problem=poisson_control,
        y0={"state": np.zeros(size), "control": np.zeros(size)},
        tags=frozenset({"equality", "bounds"}),
    )


def chained_rosenbrock(y, args, state):
    """The chained Rosenbrock function with trigonometric-exponential constraints.

    This is problem 5.1 of Lukšan and Vlček's collection of sparse test problems.
    Every constraint links three neighbouring variables, and all variables equal
    to one is the solution.
    """
    objective = jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2)
    first, second, third = y[:-2], y[1:-1], y[2:]
    constraints = (
        3.0 * second**3
        + 2.0 * third
        - 5.0
        + jnp.sin(second - third) * jnp.sin(second + third)
        + 4.0 * second
        - first * jnp.exp(first - second)
        - 3.0
    )
    return alx.Evaluation(objective=objective, equality=constraints)


def _rosenbrock_problem(size: int) -> BenchmarkProblem:
    """Return the chained Rosenbrock problem with `size` variables."""
    y0 = np.where(np.arange(size) % 2 == 0, -1.2, 1.0)
    return BenchmarkProblem(
        name=f"rosenbrock_{size}",
        problem=chained_rosenbrock,
        y0=y0,
        solution=np.ones(size),
        objective=0.0,
        tags=frozenset({"equality"}),
    )


PROBLEMS = [
    factory(size)
    for factory in (
        _simplex_problem,
        _pendulum_problem,
        _poisson_problem,
        _rosenbrock_problem,
    )
    for size in SIZES
]
"""Every family at every size in `SIZES`."""
