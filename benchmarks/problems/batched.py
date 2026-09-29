"""Families of small problems that differ only in their `args`, for `jax.vmap`."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

import aulax as alx

from .scaling import PENDULUM_MAXIMUM_TORQUE, pendulum_dynamics


@dataclass(frozen=True)
class BatchedFamily:
    """A problem function with a shared starting point and random `args`."""

    name: str
    """The name the batched benchmark uses."""
    problem: alx.ProblemFunction
    """The problem function, which takes one member's `args`."""
    y0: PyTree[np.ndarray]
    """The starting point, the same for every member of a batch."""
    sample_args: Callable[[np.random.Generator, int], PyTree[np.ndarray]]
    """Returns the `args` of a batch of the given size, with a leading batch axis."""


def shifted_hs71(y, args, state):
    """HS71 with the sum of squares constrained to `args` instead of 40."""
    return alx.Evaluation(
        objective=y[0] * y[3] * (y[0] + y[1] + y[2]) + y[2],
        equality=jnp.sum(y**2) - args,
        inequality=jnp.concatenate([jnp.array([25.0 - jnp.prod(y)]), 1.0 - y, y - 5.0]),
    )


STABILIZATION_HORIZON = 2.0
"""The time in which the pendulum has to reach the upright position, in seconds."""

STABILIZATION_STEPS = 20
"""The number of time steps in the stabilization problem."""


def pendulum_stabilization(y, args, state):
    """Bring a pendulum from the state in `args` to rest upright, as in one MPC step.

    `args` holds the initial angle and angular velocity. The objective penalizes
    the distance from upright along the way and the control effort.
    """
    theta, omega, torque = y["theta"], y["omega"], y["torque"]
    step = STABILIZATION_HORIZON / STABILIZATION_STEPS
    initial = jnp.array([theta[0] - args[0], omega[0] - args[1]])
    final = jnp.array([theta[-1] - jnp.pi, omega[-1]])
    return alx.Evaluation(
        objective=step
        * (
            jnp.sum((theta - jnp.pi) ** 2)
            + 0.1 * jnp.sum(omega**2)
            + 0.01 * jnp.sum(torque**2)
        ),
        equality={
            "dynamics": pendulum_dynamics(theta, omega, torque, step),
            "initial": initial,
            "final": final,
        },
        inequality={
            "lower": -PENDULUM_MAXIMUM_TORQUE - torque,
            "upper": torque - PENDULUM_MAXIMUM_TORQUE,
        },
    )


FAMILIES = [
    BatchedFamily(
        name="shifted_hs71",
        problem=shifted_hs71,
        y0=np.array([1.0, 5.0, 5.0, 1.0]),
        sample_args=lambda rng, size: rng.uniform(36.0, 44.0, size),
    ),
    BatchedFamily(
        name="pendulum_stabilization",
        problem=pendulum_stabilization,
        y0={
            "theta": np.full(STABILIZATION_STEPS + 1, np.pi),
            "omega": np.zeros(STABILIZATION_STEPS + 1),
            "torque": np.zeros(STABILIZATION_STEPS),
        },
        sample_args=lambda rng, size: np.stack(
            [np.pi + rng.uniform(-0.5, 0.5, size), rng.uniform(-1.0, 1.0, size)],
            axis=1,
        ),
    ),
]
"""The batched families."""
