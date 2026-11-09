# Inner solvers

Every outer iteration of the augmented Lagrangian method minimizes a smooth function with an inner minimiser. The default is `alx.Newton`, which takes Newton steps with the exact Hessian inside a trust region. It has three modes for its linear algebra. This page describes how a mode is selected automatically and how to select one explicitly.

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

With no argument, `alx.Newton()` uses `alx.AutoHessian`. It selects a mode from the size and structure of the problem when the solver is traced:

1. Problems with at most 400 variables use dense mode. Dense mode computes the inertia of the Hessian from its eigendecomposition, which makes the trust-region steps reliable on stiff problems.
2. For larger problems, the sparsity pattern of the Hessian is detected with [asdex](https://github.com/adrhill/asdex), and sparse mode is selected when fewer than 5 percent of the entries are nonzero and the coloring needs fewer than a tenth of the variables as colors.
3. Everything else uses matrix-free mode. That includes a missing `asdex` or `splineax` installation, a function that `asdex` cannot trace, and a Hessian with dense rows, which occurs when a constraint involves every variable.

A warning names the reason when sparse mode is rejected. The limits are fields of `AutoHessian`.

<!-- name: test_inner_solvers -->
```python
solver = alx.AugmentedLagrangian()
solution = alx.minimise(chain, solver, jnp.zeros(20), target)
assert solution.stats.infeasibility <= 1e-6
```

## Explicit mode selection

Pass the strategy to `alx.Newton` to override the choice:

<!-- name: test_inner_solvers -->
```python
dense = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.DenseHessian()))
free = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.MatrixFreeHessian()))

first = alx.minimise(chain, dense, jnp.zeros(20), target)
second = alx.minimise(chain, free, jnp.zeros(20), target, max_steps=4096)
assert jnp.allclose(first.value, second.value, atol=1e-4)
```

Dense mode computes the step from an eigendecomposition of the Hessian up to `eigendecomposition_limit` variables, and from Cholesky factorizations beyond that. Both handle a Hessian that is not positive definite, which happens often away from a solution.

## Sparse mode

Sparse mode needs the optional packages `asdex` and `splineax`, which `pip install aulax[sparse]` installs. It computes the Hessian with as many Hessian-vector products as the coloring of its sparsity pattern needs, and solves the Newton system with a sparse direct solver. The pattern is detected by default. The sparse direct solver does not report the inertia of the Hessian, so the step inside the trust region is computed from factorizations of the Hessian plus a multiple of the identity. On stiff problems of 200 to 300 variables in the benchmarks, dense mode converged where sparse mode needed many more steps or failed, which is why the automatic choice prefers dense mode up to 400 variables.

```python
sparse = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.SparseHessian()))
```

Detection traces the problem function, so it fails on callbacks. Top-level `jax.debug.callback` calls are the exception, because they are removed before the trace. It can also be slow for very large problems. When you know the structure, give it as a `ProblemSparsity` and nothing is traced:

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

The two patterns are those of the objective and constraint functions, and the pattern of the augmented Lagrangian is derived from them:

- `objective_hessian` is the structural pattern of the Hessian of the objective. The Hessian of the Lagrangian also works.
- `constraint_jacobian` is the structural pattern of the Jacobian of the equality residuals followed by the inequality residuals. The rows are in the order of `jax.tree.leaves((equality, inequality))` and the columns in the order of `jax.flatten_util.ravel_pytree(y)`.

A pattern is structural when an entry means that the output depends on the input, whatever the value at one point. A pair of index arrays, a `BCOO` matrix, an `asdex.SparsityPattern` or a dense array are all accepted. The constraint Hessians need no pattern of their own, because a constraint can only have a second derivative with respect to variables that appear in its row of the Jacobian, and the penalty term already contains the pattern of $J^\top J$.

A pattern that misses an entry gives a wrong Hessian without any error. `SparseHessian(check_pattern=True)` compares it with a dense Hessian at every step and raises on a difference, which is slow and intended for tests.

## Matrix-free mode

Matrix-free mode never forms the Hessian. Each step runs conjugate gradients inside the trust region with Hessian-vector products, and stops at the boundary or at a direction of negative curvature. The number of iterations grows with the condition number of the Hessian, which the penalty makes large, so a preconditioner reduces the number of iterations on hard problems. Without one it can take thousands of products per step.

`preconditioner` takes an `alx.Preconditioner`, or a plain function that builds one. The function is called with the Hessian-vector product and the current point, and returns a function that applies an approximate inverse Hessian to a vector. The approximation must be symmetric and positive definite.

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

`alx.JacobiPreconditioner` implements this preconditioner, with the diagonal either exact or estimated from random probes (`probes=16`), and `alx.IdentityPreconditioner` is the default. A Jacobi preconditioner is not always an improvement, so measure on your problem.

In matrix-free mode, each inner solve stops when the largest element of the gradient of the augmented Lagrangian is below the inner tolerance. The default `inner_stopping=None` selects this rule automatically.

## Other minimisers

Any optimistix minimiser can still do the inner solves, for example `optx.BFGS` or `optx.LBFGS`. They need only first derivatives, which matters for functions that cannot be differentiated twice. The stationarity check is then off by default, since their stopping tests usually end the inner solve before the gradient of the Lagrangian is that small. Set `stationarity_tolerance` to turn it on.
