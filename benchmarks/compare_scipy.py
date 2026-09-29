"""Compare aulax with SciPy's constrained minimizers on Hock-Schittkowski problems.

Run with `uv run --group bench python benchmarks/compare_scipy.py`. SciPy calls
jitted JAX functions for the objective, constraints and their derivatives, so
its timings include the overhead of a JAX call per evaluation.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import scipy.optimize

import aulax as alx


jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from problems import PROBLEMS, KnownProblem  # noqa: E402


SOLVER = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-10, atol=1e-10))
REPEATS = 5


def flat_constraint(tree) -> jax.Array:
    """Return the leaves of a non-empty constraint pytree as one flat array."""
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)])


def best_time(function) -> float:
    """Return the fastest of several runs of `function`, in seconds."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def run_aulax(known: KnownProblem) -> dict:
    """Solve with aulax after compiling, and return error, steps and time."""
    y0 = jnp.asarray(known.y0)
    solve = jax.jit(lambda y0: alx.minimise(known.problem, SOLVER, y0, throw=False))
    solution = jax.block_until_ready(solve(y0))
    seconds = best_time(lambda: jax.block_until_ready(solve(y0)))
    return {
        "error": float(np.max(np.abs(solution.value - known.solution))),
        "evaluations": int(solution.stats.num_steps) + 2,
        "seconds": seconds,
    }


def run_scipy(known: KnownProblem, method: str) -> dict:
    """Solve with a SciPy method, and return error, evaluations and time."""

    def evaluate(y):
        return known.problem(y, None, None)

    objective = jax.jit(lambda y: evaluate(y).objective)
    gradient = jax.jit(jax.grad(lambda y: evaluate(y).objective))
    constraints = []
    evaluation = evaluate(jnp.asarray(known.y0))
    if evaluation.equality is not None:
        equality = jax.jit(lambda y: flat_constraint(evaluate(y).equality))
        equality_jacobian = jax.jit(jax.jacfwd(equality))
        constraints.append(
            {
                "type": "eq",
                "fun": lambda y: np.asarray(equality(y)),
                "jac": lambda y: np.asarray(equality_jacobian(y)),
            }
        )
    if evaluation.inequality is not None:
        # SciPy writes inequality constraints as c(y) >= 0.
        inequality = jax.jit(lambda y: -flat_constraint(evaluate(y).inequality))
        inequality_jacobian = jax.jit(jax.jacfwd(inequality))
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y: np.asarray(inequality(y)),
                "jac": lambda y: np.asarray(inequality_jacobian(y)),
            }
        )

    def solve():
        return scipy.optimize.minimize(
            lambda y: float(objective(y)),
            known.y0,
            jac=lambda y: np.asarray(gradient(y)),
            method=method,
            constraints=constraints,
            tol=1e-10,
            options={"maxiter": 5000},
        )

    result = solve()
    return {
        "error": float(np.max(np.abs(result.x - known.solution))),
        "evaluations": int(result.nfev),
        "seconds": best_time(solve),
    }


def main():
    """Print a table with one row per problem and method."""
    print(f"{'problem':8} {'method':13} {'error':>10} {'evals':>6} {'ms':>8}")
    for known in PROBLEMS:
        rows = {
            "aulax": run_aulax(known),
            "SLSQP": run_scipy(known, "SLSQP"),
            "trust-constr": run_scipy(known, "trust-constr"),
        }
        for method, row in rows.items():
            print(
                f"{known.name:8} {method:13} {row['error']:10.1e} "
                f"{row['evaluations']:6d} {1000 * row['seconds']:8.2f}"
            )


if __name__ == "__main__":
    main()
