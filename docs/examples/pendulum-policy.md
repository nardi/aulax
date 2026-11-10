# Pendulum swing-up policy

This example trains a small neural network to swing a pendulum up from many different starting states. The network is a policy, which maps the state of the pendulum to a torque. aulax is used in two ways. It computes the optimal trajectory from each starting state, which serves as training data. It also trains the network itself, with the network weights as the variables `y` and with a constraint for every starting state that the pendulum ends upright.

The second use is what this example is about. The objective averages the control effort over all starting states, and the constraints hold for each starting state separately. Each of those constraints has its own multiplier, and the multiplier update changes it in proportion to the final error from that starting state. The starting states that end furthest from upright therefore have the largest terms in the next augmented Lagrangian.

## Pendulum model

The pendulum has an angle $\theta$, measured from hanging straight down, and an angular velocity $\omega$. A torque $u$ with $\lvert u \rvert \le 4$ drives it. The motion follows $\ddot\theta = u - g \sin\theta$, which we discretize with explicit Euler steps.

<!-- name: test_pendulum_policy -->
```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)

GRAVITY = 9.81
"""The gravitational acceleration divided by the length of the pendulum."""
MAXIMUM_TORQUE = 4.0
"""The largest torque, too small to lift the pendulum in one swing."""
HORIZON = 5.0
"""The time in which the pendulum has to reach the upright position."""
STEPS = 30
"""The number of explicit Euler steps over the horizon."""
STEP = HORIZON / STEPS
"""The length of one time step."""


def pendulum_step(theta, omega, torque):
    """Return the angle and angular velocity after one explicit Euler step."""
    return theta + STEP * omega, omega + STEP * (torque - GRAVITY * jnp.sin(theta))
```

## Optimal trajectories

For one starting state, the swing-up with the least control effort is a standard trajectory optimization problem. The angles, angular velocities and torques at every time step are the variables, and the dynamics are equality constraints between neighbouring time steps. The starting state is passed as `args`.

<!-- name: test_pendulum_policy -->
```python
def swing_up(y, initial_state, state):
    """Return the effort, the dynamics and boundary residuals, and the torque bounds."""
    theta, omega, torque = y["theta"], y["omega"], y["torque"]
    next_theta, next_omega = pendulum_step(theta[:-1], omega[:-1], torque)
    return alx.Evaluation(
        objective=STEP * jnp.sum(torque**2),
        equality={
            "dynamics": jnp.concatenate(
                [theta[1:] - next_theta, omega[1:] - next_omega]
            ),
            "start": jnp.array([theta[0], omega[0]]) - initial_state,
            "end": jnp.array([theta[-1] - jnp.pi, omega[-1]]),
        },
        inequality=jnp.abs(torque) - MAXIMUM_TORQUE,
    )


def trajectory_guess(initial_state):
    """Return a trajectory that moves at constant speed from the start to upright."""
    return {
        "theta": jnp.linspace(initial_state[0], jnp.pi, STEPS + 1),
        "omega": jnp.full(STEPS + 1, jnp.pi / HORIZON),
        "torque": jnp.zeros(STEPS),
    }


trajectory_solver = alx.AugmentedLagrangian(
    penalty_update=alx.AlgencanPenaltyUpdate(per_constraint=True)
)


@jax.jit
@jax.vmap
def optimal_trajectories(initial_state):
    """Return the optimal trajectory from every starting state in a batch."""
    solution = alx.minimise(
        swing_up,
        trajectory_solver,
        trajectory_guess(initial_state),
        initial_state,
        max_steps=10_000,
    )
    return solution.value
```

With a single penalty for all constraints, a few of these solves end at a point that is not feasible. A separate penalty for every constraint element, from `per_constraint=True`, lets the penalty grow only on the residuals that do not decrease, and then every solve succeeds.

The starting states are near the bottom, with a small positive angle and angular velocity. Starting states on both sides of the bottom would swing up in opposite directions, and a smooth network cannot switch between the two. We draw 32 starting states for training and 32 others for testing.

<!-- name: test_pendulum_policy -->
```python
def initial_states(key, count):
    """Return `count` starting angles and angular velocities near the bottom."""
    return jax.random.uniform(
        key, (count, 2), minval=jnp.array([0.1, 0.0]), maxval=jnp.array([0.6, 0.5])
    )


training_states = initial_states(jax.random.key(1), 32)
test_states = initial_states(jax.random.key(2), 32)
references = optimal_trajectories(training_states)
test_references = optimal_trajectories(test_states)
```

## Policy network

The policy is a multilayer perceptron from Equinox. Its inputs are $\cos\theta$, $\sin\theta$, $\omega$ and the fraction of the horizon that has passed, and its output passes through $\tanh$ scaled to the largest torque. The torque bound therefore holds by construction and needs no constraint.

`eqx.partition` splits the network into its weights, which are the variables of the training problems, and the rest of its structure, which the solves do not change.

<!-- name: test_pendulum_policy -->
```python
network = eqx.nn.MLP(
    4, 1, width_size=16, depth=2, activation=jnp.tanh, key=jax.random.key(0)
)
weights, network_structure = eqx.partition(network, eqx.is_array)
times = STEP * jnp.arange(STEPS)


def policy_torque(network, theta, omega, time):
    """Return the torque that the policy chooses in a state at a time."""
    features = jnp.stack([jnp.cos(theta), jnp.sin(theta), omega, time / HORIZON])
    return MAXIMUM_TORQUE * jnp.tanh(network(features)[0])
```

## Imitation of the optimal trajectories

A network with random weights is far from swinging the pendulum up, and training it directly against the terminal constraints does not converge. We first fit it to the optimal trajectories by least squares, so that it predicts the optimal torque in the states along them. This problem has no constraints, and aulax solves it like any other.

<!-- name: test_pendulum_policy -->
```python
def imitation(weights, trajectories, state):
    """Return the mean squared difference from the optimal torques."""
    network = eqx.combine(weights, network_structure)
    # The inner `vmap` runs over the time steps of one trajectory, and the outer
    # one over the trajectories.
    torques_along = jax.vmap(policy_torque, in_axes=(None, 0, 0, 0))
    predicted = jax.vmap(torques_along, in_axes=(None, 0, 0, None))(
        network, trajectories["theta"][:, :-1], trajectories["omega"][:, :-1], times
    )
    return alx.Evaluation(objective=jnp.mean((predicted - trajectories["torque"]) ** 2))


imitation_solver = alx.AugmentedLagrangian(inner=optx.LBFGS(rtol=1e-8, atol=1e-8))
imitated = alx.minimise(imitation, imitation_solver, weights, references, max_steps=5_000)
```

## Training through the simulation

The imitated policy makes small errors in every step, and they add up along a trajectory. To correct that, we simulate the pendulum under the policy and differentiate through the simulation. The objective is the mean control effort over the training starting states. The equality constraints require every trajectory to end upright, with zero angular velocity. They have the shape `(32, 2)`, so there is a multiplier for every starting state and both final values.

<!-- name: test_pendulum_policy -->
```python
def rollout(network, initial_state):
    """Simulate the pendulum under the policy and return the final state and torques."""

    def time_step(pendulum_state, time):
        theta, omega = pendulum_state
        torque = policy_torque(network, theta, omega, time)
        return pendulum_step(theta, omega, torque), torque

    final_state, torques = jax.lax.scan(
        time_step, (initial_state[0], initial_state[1]), times
    )
    return jnp.stack(final_state), torques


def policy_training(weights, initial_states, state):
    """Return the mean effort and the final state errors of the policy."""
    network = eqx.combine(weights, network_structure)
    final_states, torques = jax.vmap(rollout, in_axes=(None, 0))(
        network, initial_states
    )
    return alx.Evaluation(
        objective=STEP * jnp.mean(jnp.sum(torques**2, axis=1)),
        equality=final_states - jnp.array([jnp.pi, 0.0]),
    )


policy_solver = alx.AugmentedLagrangian(
    inner=optx.LBFGS(rtol=1e-10, atol=1e-10),
    inner_max_steps=4096,
    tolerance_schedule=alx.GeometricToleranceSchedule(initial=1e-8),
    scaling=None,
    penalty_update=alx.AlgencanPenaltyUpdate(initial_value=10.0),
)
trained = alx.minimise(
    policy_training,
    policy_solver,
    imitated.value,
    training_states,
    max_steps=20_000,
    throw=False,
)
```

The network has 369 weights, so a dense Hessian would be affordable, but every Hessian evaluation differentiates twice through 32 simulations. `optx.LBFGS` only needs gradients. Its settings are the ones that worked best for L-BFGS on the benchmark problems, which are described in the [inner solvers](../guide/inner-solvers.md#other-minimisers) guide.

A network can rarely end every trajectory exactly upright, so the training runs until `max_steps` and `throw=False` keeps the result. This is the usual way to end the training of a network, by a budget instead of a tolerance.

## Results on the test starting states

We compare the policies on the 32 test starting states, which neither training problem has seen. The final error is the largest deviation of a final state from upright and at rest, and the cost ratio is the mean of the policy's effort divided by the optimal effort for the same starting state.

<!-- name: test_pendulum_policy -->
```python
def test_performance(weights):
    """Return the largest final error and the mean effort relative to the optimum."""
    network = eqx.combine(weights, network_structure)
    final_states, torques = jax.vmap(rollout, in_axes=(None, 0))(network, test_states)
    final_error = jnp.max(jnp.abs(final_states - jnp.array([jnp.pi, 0.0])))
    effort = jnp.sum(torques**2, axis=1)
    optimal_effort = jnp.sum(test_references["torque"] ** 2, axis=1)
    return final_error, jnp.mean(effort / optimal_effort)


imitated_error, imitated_cost_ratio = test_performance(imitated.value)
trained_error, trained_cost_ratio = test_performance(trained.value)
assert trained_error < 0.05 < imitated_error
assert trained_cost_ratio < 1.05
```

| policy | largest final error | cost ratio |
|---|---|---|
| imitated | 0.26 | 1.00 |
| trained through the simulation | 0.025 | 1.00 |

The imitated policy already uses the optimal effort on average, but its errors accumulate to a final error of 0.26. Training through the simulation reduces the final error tenfold, at the same effort.

## Extensions

- **Exact constraints at run time.** The simulated trajectory of the policy can replace `trajectory_guess` as the starting point of a `swing_up` solve from the same state. The result of that solve meets the constraints to the solver's tolerance.
- **Many starting states.** With more starting states than fit in one evaluation, the training loop has to use minibatches. A stochastic optimizer from optax then takes the place of `alx.minimise`, and the loop can update the multipliers and penalties with `alx.SafeguardedMultiplierUpdate` and `alx.AlgencanPenaltyUpdate` between epochs.
- **Long horizons.** Gradients through a long simulation can grow very large. The horizon can be split into segments, with the state at the start of each segment as a variable and an equality constraint between the end of one segment and the start of the next. The gradients are then taken through short simulations only.
