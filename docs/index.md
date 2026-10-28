# aulax

aulax is a JAX library for constrained nonlinear optimization. It solves problems of the form

$$
\min_y f(y) \quad \text{subject to} \quad h(y) = 0, \quad g(y) \le 0
$$

with the augmented Lagrangian method, using the unconstrained minimisers from [optimistix](https://github.com/patrick-kidger/optimistix) for the inner solves. In other words, it lets you add equality and inequality constraints to an optimistix minimization, and keep `jax.jit`, `jax.vmap` and `jax.grad` working on the result.

Install it with pip:

```bash
pip install aulax
```

The conventional import style is `import aulax as alx`, which matches `import optimistix as optx` and `import lineax as lx`. A problem is one function that returns the objective and the constraints together:

<!-- name: test_home -->
```python
import jax.numpy as jnp
import optimistix as optx

import aulax as alx


def problem(y, args, state):
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2),
        equality=jnp.sum(y**2) - 1.0,
        inequality=y[1] - 0.3,
    )


solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-6, atol=1e-6))
solution = alx.minimise(problem, solver, jnp.zeros(2), jnp.array([2.0, 1.0]))
assert jnp.allclose(solution.value[1], 0.3, atol=1e-3)
```

The [basic usage](guide/basic-usage.md) page walks through this example.

## Why aulax?

If you have a JAX model and want to optimize it subject to constraints, the options are thin. SciPy's `minimize` has constrained methods, but it runs on the CPU in NumPy, one problem at a time, and you can't differentiate through it. optimistix has well-built minimisers, but they are unconstrained. Constrained optimization has been an [open optimistix issue](https://github.com/patrick-kidger/optimistix/issues/45) since 2024.

Several JAX libraries cover part of this, and each made different choices than aulax:

1. [jaxipm](https://github.com/johnviljoen/jaxipm) reimplements the IPOPT interior point method and is aimed at batches of small problems on one GPU. An interior point method factors a KKT matrix on every iteration. aulax only needs gradients, so its memory stays proportional to the number of variables and constraints, and it also works for a single problem too large to factor.
2. [jaxls](https://github.com/brentyi/jaxls) solves sparse nonlinear least-squares problems, like factor graphs, with Levenberg-Marquardt and an augmented Lagrangian for constraints. aulax takes a general objective instead of a sum of squares.
3. [slsqp-jax](https://pypi.org/project/slsqp-jax/) implements SLSQP with a dense quasi-Newton Hessian, which suits small problems.

aulax is also built around one function that returns everything at once. The objective and the constraints of a real model usually share most of their work, like a simulation whose output feeds both. With one function, that work runs once per point. The function can also pass a warm start to its next evaluation through its `state`, which matters when the model contains an iterative solve of its own.

The augmented Lagrangian method has its costs. It needs more function evaluations than SQP or an interior point method on small problems, and it slows down on problems with many bound constraints, because the inner minimiser handles bounds through the penalty. The [advanced usage](guide/advanced-usage.md) page shows where this bites.

## Status

aulax is in early development, and its interface may still change. It currently has:

- equality and inequality constraints on pytrees of any structure, with state threaded between evaluations,
- the ALGENCAN multiplier and penalty updates, with safeguards and optional per-constraint penalties,
- support for `jax.jit` and `jax.vmap`, and for float32 as well as float64,
- implicit differentiation through the KKT conditions, with a dense or matrix-free linear solver, and unrolled differentiation as an alternative,
- a trust-region `Newton` inner minimiser with dense, sparse or matrix-free Hessians, and reuse of the Hessian approximation across outer iterations for the quasi-Newton minimisers.

Several things are planned but not there yet: sparse direct linear solvers for the implicit derivatives, and a least-squares variant that uses Gauss-Newton for the inner solves.

aulax is licensed under the [Hippocratic License 3.0](https://github.com/nardi/aulax/blob/main/LICENSE.md).
