# Problem interface

Every problem in aulax is one function, called as `problem(y, args, state)`, that returns an `alx.Evaluation`. This page goes through the parts of that contract: the fields of `Evaluation`, the shapes that `y` and the constraints can take, and the `state` and `aux` outputs.

The reason for bundling everything into one function is that the objective and the constraints often share most of their work. Think of a simulation whose output feeds both the cost and the limits on it. With one function, that shared work runs once per point, and the gradient the solver needs costs one reverse pass through it.

## Evaluation fields

An `Evaluation` has five fields:

- `objective`, the scalar $f(y)$ to minimize,
- `equality`, the residuals $h(y)$ that should be zero at the solution,
- `inequality`, the residuals $g(y)$ that should be at most zero,
- `state`, which the solver passes to the next call of the problem function,
- `aux`, any extra output we want back, which the solver never reads.

Only `objective` is required. A problem without equality constraints leaves `equality` at `None`, and the same goes for the other fields.

The inequality convention is $g(y) \le 0$. A constraint of the form $a(y) \ge b$ becomes `b - a(y)`, and bounds on the variables are ordinary inequality constraints. A box $l \le y \le u$ becomes two residuals:

<!-- name: test_bounds -->
```python
import jax
import jax.numpy as jnp
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)


def bounded_quadratic(y, args, state):
    lower, upper = args
    return alx.Evaluation(
        objective=jnp.sum((y - 3.0) ** 2),
        inequality={"lower": lower - y, "upper": y - upper},
    )


solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
solution = alx.minimise(bounded_quadratic, solver, jnp.zeros(2), (0.0, 1.0))
assert jnp.allclose(solution.value, 1.0, atol=1e-5)
```

## Pytrees

The variables `y` can be any pytree of arrays, and so can each kind of constraint. The inequality constraints above are a dictionary with two entries, and the multipliers come back in the same structure:

<!-- name: test_bounds -->
```python
assert jnp.allclose(solution.multipliers.inequality["upper"], 4.0, atol=1e-3)
assert jnp.allclose(solution.multipliers.inequality["lower"], 0.0)
```

The solver never flattens these pytrees into one vector. It updates the multipliers and penalties leaf by leaf, so a problem whose variables or constraints are sharded across devices keeps that sharding.

## Problem state

Some problems contain an iterative computation of their own, like a nonlinear solve inside a simulation. Each evaluation could start that computation from scratch, but the points the solver visits are close together, so the result at the previous point is usually a much better starting guess. The `state` output exists for this. Whatever the problem function returns as `state` is passed to the next call.

As an example, say the objective depends on the root $z$ of $z^3 + z = y_1$. We find $z$ with a few Newton steps, starting from the root at the previous point:

<!-- name: test_state -->
```python
import jax
import jax.numpy as jnp
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)


def cubic_root_problem(y, args, state):
    z = state
    for _ in range(3):
        z = z - (z**3 + z - y[0]) / (3.0 * z**2 + 1.0)
    return alx.Evaluation(
        objective=(z - 1.0) ** 2 + y[1] ** 2,
        equality=y[0] + y[1] - 3.0,
        state=z,
        aux=z**3 + z - y[0],
    )


solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
solution = alx.minimise(cubic_root_problem, solver, jnp.zeros(2), state=jnp.array(0.0))
```

The first call receives the `state` we pass to `alx.minimise`. The solution carries the state and the aux output from its final evaluation, so we can check that three Newton steps were enough:

<!-- name: test_state -->
```python
assert abs(solution.aux) < 1e-10
assert jnp.allclose(solution.state**3 + solution.state, solution.value[0])
```

Three rules keep this correct, and aulax relies on all of them:

1. **Fixed structure.** Every `state` the function returns must have the same pytree structure, shapes and dtypes as the initial one. The solver runs in a single `jax.lax.while_loop`, and the state is part of what that loop carries.
2. **No effect on the outputs.** The objective and constraints must not depend on `state` beyond the tolerance of the computation it warm-starts. The inner minimiser compares function values between steps in its line search, and those comparisons only make sense if the same `y` always gives the same value.
3. **No gradient.** The solver wraps `state` in `jax.lax.stop_gradient` before every call. Differentiating through a warm start would cost memory and gives nothing useful.

The state is threaded from each point the inner minimiser accepts. A rejected trial point in a line search does not pass its state on.

## Auxiliary output

The `aux` field carries anything else we want to look at after the solve, such as diagnostics of an inner computation. Like `state`, it has to be a pytree of arrays with a fixed structure. The solver returns the `aux` from the evaluation at the solution.

## Type annotations

`alx.ProblemFunction` is a protocol that describes the signature. Any function that takes `(y, args, state)` and returns an `Evaluation` satisfies it, so there is nothing to subclass. It is only useful for annotating our own code:

<!-- name: test_protocol -->
```python
import jax
import jax.numpy as jnp

import aulax as alx


def paraboloid(y: jax.Array, args: None, state: None) -> alx.Evaluation[None, None]:
    return alx.Evaluation(objective=jnp.sum(y**2), equality=y[0] - 1.0)


problem: alx.ProblemFunction[jax.Array, None, None, None] = paraboloid
```
