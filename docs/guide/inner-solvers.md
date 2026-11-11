# Inner solvers

Every outer iteration of the augmented Lagrangian method minimizes a smooth function with an inner minimiser. The default is `alx.Newton`, which takes Newton steps with the exact Hessian inside a trust region. It has three ways of doing its linear algebra, and this page explains how it chooses and how to choose yourself. `alx.LBFGS` is the first-order alternative, for problems whose second derivatives are unavailable or too expensive. Both handle [bounds on the variables](bounds.md) themselves.

The examples use a small chained problem, where each variable is tied to its neighbour by an equality constraint.

<!-- name: test_inner_solvers -->
```python
import jax
import jax.numpy as jnp

import aulax as alx

jax.config.update("jax_enable_x64", True)


def chain(y, args, state):
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2) + 0.1 * jnp.sum(jnp.diff(y) ** 2),
        equality=y[1:] - 0.5 * y[:-1] - 0.1 * jnp.sin(y[1:]),
        inequality=jnp.concatenate([-y - 1.0, y - 2.0]),
    )


target = jnp.linspace(0.0, 1.0, 20)
```

## Modes

| Mode | Strategy | Cost per step | Suits |
|---|---|---|---|
| Dense | `alx.DenseHessian` | Memory $n^2$, work $n^3$ | Up to a few hundred variables |
| Sparse | `alx.SparseHessian` | Proportional to the number of nonzeros | Large problems whose Hessian is sparse |
| Matrix-free | `alx.MatrixFreeHessian` | A few Hessian-vector products per iteration | Large problems without a usable sparsity pattern |

The Hessian that these modes handle is the one of the augmented Lagrangian, which is the Hessian of the objective, plus multiples of the constraint Hessians, plus the penalty times $J^\top J$ for the Jacobian $J$ of the constraints.

## Automatic choice

With no argument, `alx.Newton()` uses `alx.AutoHessian`. It looks at the problem when the solver is traced:

1. Problems with at most 400 variables use dense mode. Dense mode reads the inertia of the Hessian from its eigendecomposition, and that keeps the trust-region steps reliable on stiff problems.
2. Larger problems try sparse mode. The sparsity of the Hessian is detected with [asdex](https://github.com/adrhill/asdex), and the result is accepted when fewer than 5 percent of the entries are nonzero and the coloring needs fewer than a tenth of the variables as colors.
3. Everything else uses matrix-free mode. That includes a missing `asdex` or `splineax` installation, a function that `asdex` cannot trace, and a Hessian with dense rows, which a constraint that involves every variable produces.

A warning names the reason when sparse mode is rejected. The limits are fields of `AutoHessian`.

<!-- name: test_inner_solvers -->
```python
solver = alx.AugmentedLagrangian()
solution = alx.minimise(chain, solver, jnp.zeros(20), target)
assert solution.stats.infeasibility <= 1e-6
```

## Choosing a mode

Pass the strategy to `alx.Newton` to override the choice:

<!-- name: test_inner_solvers -->
```python
dense = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.DenseHessian()))
free = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.MatrixFreeHessian()))

first = alx.minimise(chain, dense, jnp.zeros(20), target)
second = alx.minimise(chain, free, jnp.zeros(20), target, max_steps=4096)
assert jnp.allclose(first.value, second.value, atol=1e-4)
```

Dense mode takes the step from an eigendecomposition of the Hessian up to `eigendecomposition_limit` variables, and from Cholesky factorizations beyond that. Both handle a Hessian that is not positive definite, which happens often away from a solution.

## Sparse mode

Sparse mode needs the optional packages `asdex` and `splineax`, which `pip install aulax[sparse]` installs. It computes the Hessian with as many Hessian-vector products as the coloring of its sparsity pattern needs, and solves the Newton system with a sparse direct solver. The pattern is detected by default. The sparse direct solver does not report the inertia of the Hessian, so the step inside the trust region relies on a shifted factorization. On stiff problems of 200 to 300 variables in the benchmarks, dense mode converged where sparse mode needed many more steps or failed, which is why the automatic choice prefers dense mode up to 400 variables.

```python
sparse = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.SparseHessian()))
```

Detection traces the problem function, so it cannot handle callbacks, except that top-level `jax.debug.callback` calls are dropped first. It can also be slow for very large problems. When you know the structure, give it as a `ProblemSparsity` and nothing is traced:

```python
import numpy as np

size = 20
# The equality residual i involves variables i and i + 1, and every bound
# involves one variable.
equality_pattern = np.eye(size - 1, size) + np.eye(size - 1, size, k=1)
bound_pattern = np.concatenate([np.eye(size), np.eye(size)])
jacobian_pattern = np.concatenate([equality_pattern, bound_pattern])
sparsity = alx.ProblemSparsity(
    objective_hessian=(jnp.arange(size), jnp.arange(size)),
    constraint_jacobian=jacobian_pattern,
)
sparse = alx.AugmentedLagrangian(
    inner=alx.Newton(hessian=alx.SparseHessian(sparsity=sparsity))
)
```

The two patterns describe the functions you wrote, and they are enough to derive the pattern of the augmented Lagrangian:

- `objective_hessian` is the structural pattern of the Hessian of the objective. The Hessian of the Lagrangian also works.
- `constraint_jacobian` is the structural pattern of the Jacobian of the equality residuals followed by the inequality residuals. The rows follow `jax.tree.leaves((equality, inequality))` and the columns follow `jax.flatten_util.ravel_pytree(y)`.

A pattern is structural when an entry means that the output depends on the input, whatever the value at one point. A pair of index arrays, a `BCOO` matrix, an `asdex.SparsityPattern` or a dense array all work. The constraint Hessians need no pattern of their own, because a constraint can only have a second derivative with respect to variables that appear in its row of the Jacobian, and the penalty term already adds the pattern of $J^\top J$.

A pattern that misses an entry gives a wrong Hessian without any error. `SparseHessian(check_pattern=True)` compares it with a dense Hessian at every step and raises on a difference, which is slow and meant for tests.

## Matrix-free mode

Matrix-free mode never forms the Hessian. Each step runs conjugate gradients inside the trust region with Hessian-vector products, and stops at the boundary or at a direction of negative curvature. The number of iterations grows with the condition number of the Hessian, which the penalty makes large, so a preconditioner pays off on hard problems. Without one it can take thousands of products per step.

`preconditioner` takes a `alx.Preconditioner`, or a plain function that builds one. The function receives the Hessian-vector product and the current point, and returns a function that applies an approximate inverse Hessian to a vector. The approximation must be symmetric and positive definite.

<!-- name: test_inner_solvers -->
```python
def diagonal_preconditioner(hessian_vector_product, point):
    size = point.size
    diagonal = jax.vmap(
        lambda index: hessian_vector_product(jnp.zeros(size).at[index].set(1.0))[index]
    )(jnp.arange(size))
    inverse = 1.0 / jnp.maximum(jnp.abs(diagonal), 1e-8)
    return lambda vector: inverse * vector


preconditioned = alx.AugmentedLagrangian(
    inner=alx.Newton(
        hessian=alx.MatrixFreeHessian(preconditioner=diagonal_preconditioner)
    )
)
solution = alx.minimise(chain, preconditioned, jnp.zeros(20), target, max_steps=4096)
assert solution.stats.infeasibility <= 1e-6
```

`alx.JacobiPreconditioner` does this, with the diagonal either exact or estimated from random probes (`probes=16`), and `alx.IdentityPreconditioner` is the default. A Jacobi preconditioner is not always an improvement, so measure on your problem.

Matrix-free mode stops each inner solve on the gradient of the augmented Lagrangian. The default `inner_stopping=None` selects that automatically.

## L-BFGS

`alx.LBFGS` is a limited-memory BFGS minimiser that needs only the gradient of the augmented Lagrangian. It stores the last `history_length` pairs of a step and a gradient difference, which is `2 * history_length * n` numbers for $n$ variables, and computes each search direction from them with the two-loop recursion. Each step evaluates the function and its gradient at one trial point. The trial point is accepted when the function decreased by at least $10^{-4}$ times the decrease that the gradient predicts along the step. After a rejection, the step length shrinks to the minimum of a quadratic fitted along the step. After a line search that cannot find a decrease, the history is discarded and the next step is a steepest descent step. The minimiser stops when the last accepted step changed the variables and the function value by less than `rtol` and `atol` allow, or when the projected gradient is at most `atol`. Both are $10^{-10}$ by default.

<!-- name: test_inner_solvers -->
```python
lbfgs = alx.AugmentedLagrangian(inner=alx.LBFGS())
solution = alx.minimise(chain, lbfgs, jnp.zeros(20), target, max_steps=4096)
assert solution.stats.infeasibility <= 1e-6
```

The fields of `alx.LBFGS` are:

- `history_length`, the number of secant pairs, 20 by default.
- `variable_scaling`, which is `"bounds"` by default and can be `"none"`. With `"bounds"`, a variable with two finite bounds is divided by the width of its box and all others by one. The step lengths and both stopping tests are then measured in the divided variables. Without bounds, the choice has no effect.
- `diagonal`, which is false by default. When it is true, the initial inverse Hessian of the recursion is a diagonal that is fitted to the secant pairs. Otherwise it is a multiple of the identity.
- `rtol`, `atol` and `norm`, which have the same meaning as for the optimistix minimisers.

The state of `alx.LBFGS` has the gradient in `f_info.grad` and the aux output of the accepted point in `aux`, which are the fields that `alx.OptimistixStateAccess` reads. The multipliers are therefore updated in the iteration where an inner solve ends, and `alx.GradientStopping` can use the gradient. The history is also what `hessian_reuse` of `alx.OptimistixStateAccess` copies into the next inner solve.

With `inner=alx.LBFGS()`, `alx.AugmentedLagrangian` selects these settings for the fields that are `None`:

- `penalty_update` is `alx.AlgencanPenaltyUpdate(initial_value=10.0, growth_factor=100.0)`.
- `tolerance_schedule` is `alx.GeometricToleranceSchedule(initial=1e-8)`.
- `inner_state_access` is `alx.OptimistixStateAccess(hessian_reuse="keep_unless_penalty_increased")`, so the history of the previous inner solve is kept unless the penalty increased.
- `feasibility_tolerance` is $10^{-8}$ in float64 and $10^{-5}$ in float32, which is the same as for `alx.Newton`.
- `inner_tolerance_floor` is zero in float64 and $10^{-6}$ in float32. This default depends on the dtype and not on the minimiser.

The other fields have the defaults that they have for any minimiser other than `alx.Newton`. The stationarity check is off, the penalty stops growing once the infeasibility is within 0.1 times the feasibility tolerance, and `unbounded_threshold` is $10^{20}$. The values above were chosen on the benchmark problems. On the Hock-Schittkowski problems and the S2MPJ problems with bounds, `alx.LBFGS` solves NUMBERS-PENDING.

## Bounds in the inner minimisers

`alx.Newton` and `alx.LBFGS` have the class attribute `supports_bounds`. When `alx.minimise` is called with `bounds`, they receive the box as `options["bounds"]` and run as projected variants, and every point they evaluate is inside the box. Without bounds they run as described above, and the code of the projection is not traced. The [bounds](bounds.md) page describes what this means for the solution.

`alx.Newton` is then a projected Newton method in the style of TRON. At the accepted point, a variable is fixed when it is at a bound and the negative gradient points out of the box, or when its lower and upper bound are equal. The trust-region step is computed for the free variables only, from the Hessian with the rows and columns of the fixed variables removed. The trial point is the projection of the accepted point plus this step onto the box. When the projected step is not a descent direction, the trial point is the projection of a steepest descent step with the length of the trust radius. The acceptance test and the update of the radius use the projected step, so they are the same as without bounds. The test for rounding noise compares the norms of the projected gradients.

`alx.LBFGS` clips every trial point into the box, and the search direction is computed for the free variables only. A variable is held at its bound when the gradient points out of the box. The stopping test on the gradient uses the projected gradient, and `variable_scaling` divides the variables by the width of their box. The history of secant pairs is stored in these divided variables.

Any other minimiser does not receive the box. `alx.minimise` adds the finite bounds to the inequality constraints for it, as the [bounds](bounds.md#two-ways-of-handling-bounds) page describes.

## Other minimisers

Any optimistix minimiser can still do the inner solves, for example `optx.BFGS` or `optx.LBFGS`. Like `alx.LBFGS`, they need only first derivatives, which matters for functions that cannot be differentiated twice. Unlike `alx.LBFGS`, they do not handle bounds, so `alx.minimise` adds the bounds to their inequality constraints. The stationarity check is off by default for these minimisers and for `alx.LBFGS`, since their stopping tests do not reach it. Set `stationarity_tolerance` to turn it on.

A line search can follow a direction along which the objective decreases faster than the penalty term grows, which happens for a cubic objective whose bounds are inequality constraints. `unbounded_threshold` handles this for minimisers other than `Newton`, including `alx.LBFGS`. When the augmented Lagrangian at an inner iterate is below $-10^{20}$, the solver returns to the start of the inner solve and grows the penalty.

On the Hock-Schittkowski benchmark, `optx.LBFGS` solved 46 of the 48 problems that `Newton` solved, with about 5 times as many evaluations on those problems. It needed a tight tolerance schedule, such as `GeometricToleranceSchedule(initial=1e-8)` with `rtol=atol=1e-10`, and no problem scaling. The `lbfgs_tight_penalty_10` benchmark configuration has these settings. The tolerance schedule and the initial penalty of 10 are defaults of `alx.LBFGS`, which is the minimiser to use when the inner solves need only first derivatives.
