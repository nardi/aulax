# Basic usage

Suppose we want the point on the unit circle that lies closest to the point $p = (2, 1)$, but we only accept points whose second coordinate is at most $0.3$. Written as an optimization problem, that is

$$
\min_y \lVert y - p \rVert^2 \quad \text{subject to} \quad y_1^2 + y_2^2 - 1 = 0, \quad y_2 - 0.3 \le 0.
$$

The first constraint is an equality constraint $h(y) = 0$, which keeps $y$ on the circle. The second is an inequality constraint $g(y) \le 0$. Without it, the answer would be $p / \lVert p \rVert \approx (0.894, 0.447)$, so the inequality will be active at the solution.

## Problem function

In aulax, a problem is a single function that takes the variables `y`, some fixed `args` and a `state`, and returns an `alx.Evaluation`:

<!-- name: test_basic_usage -->
```python
import jax
import jax.numpy as jnp
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)


def closest_point(y, args, state):
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2),
        equality=jnp.sum(y**2) - 1.0,
        inequality=y[1] - 0.3,
    )
```

We pass $p$ through `args` instead of writing it into the function, so the same function works for any target point. We don't use `state` here. It is there for problems that want to carry information from one evaluation to the next, which the [problem interface](problem-interface.md) page covers.

## Solver

aulax does not minimize anything by itself. It turns the constrained problem into a sequence of unconstrained ones, and hands each of those to an optimistix minimiser. So the solver is an `alx.AugmentedLagrangian` wrapped around one:

<!-- name: test_basic_usage -->
```python
solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
solution = alx.minimise(closest_point, solver, jnp.zeros(2), jnp.array([2.0, 1.0]))
```

The `rtol` and `atol` of the inner minimiser are the tolerances of its last solve. The early solves use looser tolerances, because the multipliers are still poor and a precise answer would be wasted. The [theory](../theory.md) page explains how the solver moves from one to the other.

## Solution

The minimizer is in `solution.value`. The second coordinate sits on its bound of $0.3$, and the first follows from the circle:

<!-- name: test_basic_usage -->
```python
assert solution.result == alx.RESULTS.successful
assert jnp.allclose(solution.value, jnp.array([jnp.sqrt(1.0 - 0.3**2), 0.3]), atol=1e-5)
```

`alx.minimise` raises an error when the solve fails, unless we pass `throw=False`. In that case `solution.result` says what went wrong, and `alx.RESULTS[solution.result]` returns the message.

The solution also contains the Lagrange multipliers, one per constraint, in the same shape as the constraints themselves:

<!-- name: test_basic_usage -->
```python
assert jnp.allclose(solution.multipliers.equality, 1.0966, atol=1e-3)
assert jnp.allclose(solution.multipliers.inequality, 0.7421, atol=1e-3)
```

A multiplier tells us how much the optimal objective would change if we relaxed its constraint a little. Here the inequality multiplier is about $0.74$. If we allowed $y_2 \le 0.3 + \varepsilon$ instead, the squared distance would drop by roughly $0.74 \varepsilon$. An inactive inequality constraint always has a multiplier of zero, since relaxing it changes nothing.

Finally, `solution.stats` records how the solve went:

<!-- name: test_basic_usage -->
```python
stats = solution.stats
assert stats.infeasibility < 1e-6
print(stats.num_steps, stats.num_outer_steps)
```

`num_outer_steps` counts how often the solver updated the multipliers, and `num_steps` counts the inner minimiser's steps over all inner solves. `infeasibility` is the largest constraint violation at the solution, and `stationarity` is the largest element of the gradient of the Lagrangian. Both should be close to zero after a successful solve.
