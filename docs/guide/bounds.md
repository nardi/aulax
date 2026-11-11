# Bounds on the variables

A box $l \le y \le u$ on the variables is common enough that `alx.minimise` has an argument for it. The [problem interface](problem-interface.md) page shows that a box can also be written as inequality constraints. Passing it as `bounds` is different for the inner minimisers `alx.Newton` and `alx.LBFGS`, because they keep every iterate inside the box and the bounds are then not part of the augmented Lagrangian. The solution satisfies such bounds exactly, and the bounds add no one-sided penalty terms to the function that is minimized.

The examples on this page use a quadratic whose unconstrained minimum is outside the box.

<!-- name: test_bounds_guide -->
```python
import jax
import jax.numpy as jnp
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)


def distance(y, args, state):
    return alx.Evaluation(objective=jnp.sum((y - args) ** 2))


target = jnp.array([3.0, 0.5, -1.0])
bounds = alx.Bounds(lower=0.0, upper=2.0)
```

## Specifying bounds

`alx.Bounds(lower, upper)` has two fields. Each field is either `None`, which means that there are no bounds on that side, or a pytree with the structure of `y` whose leaves are scalars or arrays. A leaf is broadcast to the shape of the matching leaf of `y`, so a scalar applies to every element of that leaf. An entry of negative infinity in `lower` or of positive infinity in `upper` means that the element has no bound on that side. When `y` is an array, the fields are numbers or arrays. When `y` is a dictionary, the fields are dictionaries with the same keys. For example, the variables `{"position": ..., "torque": ...}` with limits on the torque only have these bounds:

```python
torque_limits = alx.Bounds(
    lower={"position": -jnp.inf, "torque": -4.0},
    upper={"position": jnp.inf, "torque": 4.0},
)
```

The leaves have the dtype of the matching leaves of `y`.

## Solving with bounds

The bounds are the `bounds` argument of `alx.minimise`. The target in the example has the entries $3$, $0.5$ and $-1$, and the box is $[0, 2]$. The first and last entry of the solution are therefore at the bounds, and the second is the target itself:

<!-- name: test_bounds_guide -->
```python
solver = alx.AugmentedLagrangian()
solution = alx.minimise(distance, solver, jnp.ones(3), target, bounds=bounds)
assert jnp.allclose(solution.value, jnp.array([2.0, 0.5, 0.0]))
```

The solver here is the default `alx.Newton`, which handles the bounds itself. A problem can have equality and inequality constraints in its `alx.Evaluation` and bounds at the same time.

## Two ways of handling bounds

A minimiser class declares whether it handles bounds with the class attribute `supports_bounds`. It is true for `alx.Newton` and `alx.LBFGS`.

- A minimiser with `supports_bounds` receives the bounds, expanded to the shape of `y`, as `options["bounds"]`. Every iterate that it evaluates is inside the box, and the bounds are not constraints of the augmented Lagrangian. An element at a bound is exactly equal to it. The [inner solvers](inner-solvers.md#bounds-in-the-inner-minimisers) page describes how the two minimisers do this.
- Any other minimiser, such as `optx.BFGS` or an optax method, cannot keep iterates in a box. `alx.minimise` then adds one inequality row $l - y \le 0$ for every finite lower bound and one row $y - u \le 0$ for every finite upper bound. The bounds are satisfied to the `feasibility_tolerance` of the solver, so an element can be slightly outside the box. The multipliers of these rows are not in `Solution.multipliers`, which only has the multipliers of the constraints in the problem function. They are returned as `Solution.bound_multipliers`.

Both ways return the same result type, so the code that reads the solution does not depend on the choice:

<!-- name: test_bounds_guide -->
```python
bfgs_solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
bfgs_solution = alx.minimise(distance, bfgs_solver, jnp.ones(3), target, bounds=bounds)
assert jnp.allclose(bfgs_solution.value, solution.value, atol=1e-5)
```

Passing `options` with the key `"bounds"` to `alx.minimise` raises a `ValueError` when `bounds` is given, since `bounds` sets that key.

## Multipliers of the bounds

`Solution.bound_multipliers` is `None` when no bounds were passed. Otherwise it is an `alx.Bounds` whose fields `lower` and `upper` have the shape of `Solution.value`. An element is the reduced cost of that variable. It is nonnegative, and it is zero where the bound is not active or does not exist. Where `y` is at its lower bound, the reduced cost is the positive part of the gradient of the Lagrangian, and where `y` is at its upper bound, it is the positive part of the negated gradient. The gradient of the Lagrangian here has the objective and the constraints but not the bounds, and at the solution it equals `lower - upper`.

In the example, the gradient of the objective at the solution is $(-2, 0, 2)$. The first variable would decrease the objective by growing, and the upper bound stops it. The last variable would do the same by shrinking, and the lower bound stops it.

<!-- name: test_bounds_guide -->
```python
reduced_costs = solution.bound_multipliers
assert jnp.allclose(reduced_costs.lower, jnp.array([0.0, 0.0, 2.0]))
assert jnp.allclose(reduced_costs.upper, jnp.array([2.0, 0.0, 0.0]))
```

A reduced cost says how much the optimal objective changes when its bound moves. Raising the upper bound of the first variable by $\varepsilon$ lowers the optimal objective by about $2 \varepsilon$. The multipliers are in the units of the problem as given, also when the solver scales the problem.

## Stationarity

At a variable that is at a bound, the gradient of the Lagrangian does not have to be zero, because the bound balances it. For a minimiser with `supports_bounds`, `Statistics.stationarity` is therefore the largest element of the projected gradient of the Lagrangian, which only has the part of the gradient that points into the box. At a lower bound that part is a negative gradient, and at an upper bound it is a positive gradient. The `stationarity_tolerance` of the solver is checked against the same measure. For any other minimiser, the bounds are rows of the inequality constraints, so `Statistics.stationarity` is the largest element of the gradient of the Lagrangian that includes those rows and their multipliers.

In the example, the gradient is $(-2, 0, 2)$ and every nonzero element points out of the box, so the projected gradient is zero:

<!-- name: test_bounds_guide -->
```python
assert solution.stats.stationarity < 1e-8
```

## Derivatives

`jax.grad`, `jax.jacrev` and `jax.jacfwd` work through `alx.minimise` with bounds, with respect to `args`, to arrays that the problem function closes over, and to the arrays in `bounds`. Derivatives flow through `Solution.value`, `Solution.multipliers` and `Solution.bound_multipliers`. The default `optx.ImplicitAdjoint` includes the bounds in the KKT conditions. A bound counts as active when its reduced cost is positive, and then it is treated like an active inequality constraint with the reduced cost as its multiplier. A bound that a variable touches with a reduced cost of zero counts as inactive, which is the choice that the [advanced usage](advanced-usage.md#degenerate-constraints) page describes for inequality constraints.

An active bound holds its variable, so the derivative of that element of `value` with respect to the target is zero, and the derivative with respect to the bound is one. The element between the bounds follows the target:

<!-- name: test_bounds_guide -->
```python
def bounded_value(target, upper):
    box = alx.Bounds(lower=0.0, upper=upper)
    return alx.minimise(distance, solver, jnp.ones(3), target, bounds=box).value


by_target, by_upper = jax.jacrev(bounded_value, argnums=(0, 1))(
    target, jnp.full(3, 2.0)
)
assert jnp.allclose(by_target, jnp.diag(jnp.array([0.0, 1.0, 0.0])))
assert jnp.allclose(by_upper, jnp.diag(jnp.array([1.0, 0.0, 0.0])))
```

## Starting point and invalid bounds

The starting point `y0` is moved into the box before the solve, with `alx.Bounds.clip`, which replaces each element by the nearest point of the box. A `y0` outside the box is therefore allowed:

<!-- name: test_bounds_guide -->
```python
outside = alx.minimise(distance, solver, jnp.full(3, 9.0), target, bounds=bounds)
assert jnp.allclose(outside.value, solution.value)
```

A lower bound that is above the matching upper bound makes the box empty, and a bound that is NaN is not a number. In both cases `alx.minimise` raises a `ValueError` when the bounds are concrete values. When `jax.jit` or `jax.vmap` traces the bounds, the check cannot run before the solve. It is part of the solve instead, and a violation raises a runtime error.

<!-- name: test_bounds_guide -->
```python
import pytest

empty_box = alx.Bounds(lower=3.0, upper=2.0)
with pytest.raises(ValueError, match="lower bound must be at most"):
    alx.minimise(distance, solver, jnp.ones(3), target, bounds=empty_box)
```
