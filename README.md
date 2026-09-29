# aulax

aulax is a JAX library for constrained nonlinear optimization. It minimizes an objective subject to equality constraints $h(y) = 0$ and inequality constraints $g(y) \le 0$ with the augmented Lagrangian method, and it uses the unconstrained minimisers from [optimistix](https://github.com/patrick-kidger/optimistix) for the inner solves. In other words, it lets you add constraints to an optimistix minimization, and keep `jax.jit`, `jax.vmap` and `jax.grad` working on the result.

Install with pip:

```bash
pip install aulax
```

and [read the documentation here](https://nardi.github.io/aulax). The conventional import style is `import aulax as alx`.

## Example

A problem is one function that returns the objective and all constraints together, so any work they share runs once per point:

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
print(solution.value, solution.multipliers)
```

## Why?

SciPy's constrained methods run on the CPU, one problem at a time, and you can't differentiate through them. optimistix runs on accelerators and differentiates, but it has no constraints. aulax fills that gap for general nonlinear problems:

1. It only needs gradients. It never forms a constraint Jacobian or factors a KKT matrix during the solve, so memory stays proportional to the number of variables and constraints.
2. It runs as a single `jax.lax.while_loop`, so it works under `jax.jit` and `jax.vmap`, and a batch of problems progresses without waiting on its slowest member between multiplier updates.
3. It differentiates the solution through the KKT conditions with one linear solve, whatever the number of iterations.
4. The problem function can pass a warm start to its next evaluation, which helps when the model contains an iterative solve of its own.

The method has its costs. It needs more function evaluations than SQP or interior point methods on small problems, and it slows down on problems with many bound constraints. The documentation says more about both.

## Status

aulax is in early development, and nothing in it is stable yet. Sparse direct linear solvers for the derivatives and a least-squares variant are planned.

## License

aulax is licensed under the [Hippocratic License 3.0](LICENSE.md).
