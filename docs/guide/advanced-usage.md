# Advanced usage

This page covers two topics that come up once a basic solve works: making solves fast, and differentiating through them. All examples use the same problem, the projection of a point $p$ onto the probability simplex,

$$
\min_y \lVert y - p \rVert^2 \quad \text{subject to} \quad \sum_i y_i = 1, \quad y \ge 0.
$$

<!-- name: test_advanced_usage -->
```python
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx

import aulax as alx

jax.config.update("jax_enable_x64", True)


def simplex_projection(y, args, state):
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2),
        equality=jnp.sum(y) - 1.0,
        inequality=-y,
    )


solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
```

## Performance

### Batches of problems

`alx.minimise` works under `jax.vmap`, so a batch of problems that differ in their `args` or starting points can run as one computation:

<!-- name: test_advanced_usage -->
```python
points = jnp.array([[0.6, 0.5, -0.4], [0.2, 0.3, 0.5], [2.0, 0.0, 0.0]])


@jax.jit
@jax.vmap
def project(point):
    return alx.minimise(simplex_projection, solver, jnp.full(3, 1 / 3), point).value


projections = project(points)
assert jnp.allclose(projections[0], jnp.array([0.55, 0.45, 0.0]), atol=1e-5)
assert jnp.allclose(projections[2], jnp.array([1.0, 0.0, 0.0]), atol=1e-5)
```

Every problem in the batch runs until the slowest one finishes, since the batch shares one loop. Within that loop, each problem updates its own multipliers whenever its own inner solve is done, without waiting for the others. The [theory](../theory.md#the-solver-loop) page explains why the loop is built that way.

### Precision

Tests and examples on this site run in float64, which we enable with `jax.config.update("jax_enable_x64", True)`. In float32, a minimiser can only locate $y$ to about $3 \times 10^{-4}$ relative accuracy, and the default feasibility tolerance is $10^{-3}$ to match. The penalty cap and multiplier bounds also have lower float32 defaults, since large penalties and multipliers lose most of their significant digits there. A float32 solve looks like this:

<!-- name: test_advanced_usage -->
```python
float32_solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-5, atol=1e-5))
point = jnp.array([0.6, 0.5, -0.4], dtype=jnp.float32)
solution = alx.minimise(
    simplex_projection, float32_solver, jnp.full(3, 1 / 3, jnp.float32), point
)
assert solution.value.dtype == jnp.float32
assert jnp.allclose(solution.value, jnp.array([0.55, 0.45, 0.0]), atol=1e-3)
```

### Inner minimiser

Any optimistix minimiser can do the inner solves. `optx.BFGS` stores a dense Hessian approximation, which costs $n^2$ memory for $n$ variables and $n^2$ work per step. For more than a few hundred variables, `optx.LBFGS` keeps only a few recent gradient differences and is usually the better choice:

<!-- name: test_advanced_usage -->
```python
large_solver = alx.AugmentedLagrangian(inner=optx.LBFGS(rtol=1e-8, atol=1e-8))
point = jnp.linspace(-1.0, 1.0, 100)
solution = alx.minimise(
    simplex_projection, large_solver, jnp.full(100, 0.01), point, max_steps=4096
)
assert jnp.allclose(jnp.sum(solution.value), 1.0, atol=1e-6)
```

The inequality terms of the augmented Lagrangian have a kink in their second derivative, and quasi-Newton minimisers slow down when many constraints cross it during a solve. This 100-variable projection takes over a thousand inner steps, more than the default `max_steps` allows. That is far more than a method that handles bounds directly would need, and it is the main weakness of aulax on problems with many bound constraints.

### Initial penalty

The default initial penalty scales with the objective at `y0`. For the projection above, the objective is large while `y0` is feasible, and the formula starts with a penalty of about 340, which makes the first inner problems badly conditioned. A fixed initial penalty of 1 cuts the number of steps by a factor of four here:

<!-- name: test_advanced_usage -->
```python
default_steps = solution.stats.num_steps
gentle_solver = alx.AugmentedLagrangian(
    inner=optx.LBFGS(rtol=1e-8, atol=1e-8),
    penalty_update=alx.AlgencanPenaltyUpdate(initial_value=1.0),
)
gentle = alx.minimise(simplex_projection, gentle_solver, jnp.full(100, 0.01), point)
assert gentle.stats.num_steps < default_steps
```

When constraints have very different scales, `AlgencanPenaltyUpdate(per_constraint=True)` gives each constraint element its own penalty, so a hard constraint does not raise the penalty on all the others. Scaling the residuals in the problem function, so that a violation of 1 means roughly the same thing for every constraint, helps just as much and costs nothing.

### Warm starts

A sequence of nearby problems, like the steps of a model predictive controller, can start each solve from the previous solution. `alx.minimise` accepts the previous multipliers through `multipliers`, and the previous `state` through `state`:

<!-- name: test_advanced_usage -->
```python
first = alx.minimise(simplex_projection, gentle_solver, jnp.full(100, 0.01), point)
shifted_point = point + 0.01
cold = alx.minimise(
    simplex_projection, gentle_solver, jnp.full(100, 0.01), shifted_point
)
warm = alx.minimise(
    simplex_projection,
    gentle_solver,
    first.value,
    shifted_point,
    first.state,
    multipliers=first.multipliers,
)
assert jnp.allclose(warm.value, cold.value, atol=1e-5)
assert warm.stats.num_steps < cold.stats.num_steps
```

Here the warm start takes about a third of the steps of the cold one. The penalty always restarts from its initial value, so a warm start saves the most when the initial penalty is already about right.

## Differentiation

### Implicit derivatives

The solution of a problem depends on its `args`, and on any arrays the problem function closes over. `jax.grad`, `jax.jacrev` and `jax.jacfwd` all work on `alx.minimise`, and by default they use the implicit function theorem on the KKT conditions at the solution. This costs one linear solve, whatever the number of steps the solve itself took, and it never stores the iterations.

For the simplex projection, the Jacobian of the projection with respect to $p$ has a closed form. On the set $S$ of positive entries, the projection subtracts the mean excess, so its Jacobian there is $I - \frac{1}{\lvert S \rvert} \mathbf{1}\mathbf{1}^\top$, and it is zero elsewhere:

<!-- name: test_advanced_usage -->
```python
def project_one(point, adjoint=optx.ImplicitAdjoint()):
    return alx.minimise(
        simplex_projection, solver, jnp.full(3, 1 / 3), point, adjoint=adjoint
    ).value


point = jnp.array([0.6, 0.5, -0.4])
jacobian = jax.jacrev(project_one)(point)
expected = jnp.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
assert jnp.allclose(jacobian, expected, atol=1e-6)
```

Derivatives flow through `Solution.value` and `Solution.multipliers`. The `state`, `aux` and `stats` of a solution are not differentiable, and taking a gradient through them raises an error.

### Linear solver

The default `optx.ImplicitAdjoint()` builds the KKT matrix explicitly and factors it, which is exact and robust but costs $(n + m)^2$ memory for $n$ variables and $m$ constraints. For large problems, a Krylov solver from lineax only needs products with the matrix, which cost about one gradient evaluation each:

<!-- name: test_advanced_usage -->
```python
matrix_free = optx.ImplicitAdjoint(linear_solver=lx.GMRES(rtol=1e-10, atol=1e-10))
jacobian = jax.jacrev(project_one)(point, matrix_free)
assert jnp.allclose(jacobian, expected, atol=1e-6)
```

The KKT matrix is symmetric but indefinite, so conjugate gradients do not apply. GMRES works without a preconditioner on well-scaled problems, but it can stall on badly conditioned ones and then raises an error during the backward pass.

### Unrolled derivatives

`optx.RecursiveCheckpointAdjoint()` differentiates through every step of the solve instead. It needs memory that grows with the number of steps, and its result depends on how far the solve converged, but it also works when the KKT conditions are degenerate:

<!-- name: test_advanced_usage -->
```python
unrolled = jax.jacrev(project_one)(point, optx.RecursiveCheckpointAdjoint())
assert jnp.allclose(unrolled, expected, atol=1e-4)
```

### Degenerate constraints

The implicit derivative counts an inequality constraint as active when its multiplier is positive. A constraint that holds with equality at the solution but has a zero multiplier sits exactly on the boundary between the two cases. The derivative is not unique there, and aulax treats the constraint as inactive, which gives one of the valid one-sided derivatives.
