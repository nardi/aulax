"""Time batches of small problems solved under `jax.vmap`.

Run with `uv run --group bench python benchmarks/batched.py`. Each problem is
HS71 with its equality constraint moved to a different right-hand side, so the
problems in a batch have different solutions and take different numbers of steps.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

import aulax as alx


jax.config.update("jax_enable_x64", True)

SOLVER = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-10, atol=1e-10))
BATCH_SIZES = [1, 10, 100, 1000]


def shifted_hs71(y, args, state):
    """HS71 with the sum of squares constrained to `args` instead of 40."""
    return alx.Evaluation(
        objective=y[0] * y[3] * (y[0] + y[1] + y[2]) + y[2],
        equality=jnp.sum(y**2) - args,
        inequality=jnp.concatenate([jnp.array([25.0 - jnp.prod(y)]), 1.0 - y, y - 5.0]),
    )


@jax.jit
def solve_batch(targets):
    """Solve one problem per target under `jax.vmap`."""
    y0 = jnp.array([1.0, 5.0, 5.0, 1.0])
    return jax.vmap(
        lambda target: alx.minimise(shifted_hs71, SOLVER, y0, target, throw=False)
    )(targets)


def seconds_for(targets) -> float:
    """Return the fastest of three timed runs, after one compiling run."""
    jax.block_until_ready(solve_batch(targets))
    times = []
    for _ in range(3):
        start = time.perf_counter()
        jax.block_until_ready(solve_batch(targets))
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    """Print the time per batch and per problem for several batch sizes."""
    print(f"device: {jax.devices()[0].device_kind}")
    print(
        f"{'batch':>6} {'ms':>9} {'ms/problem':>11} {'max steps':>10} {'min steps':>10}"
    )
    rng = np.random.default_rng(0)
    for batch_size in BATCH_SIZES:
        targets = jnp.asarray(rng.uniform(36.0, 44.0, batch_size))
        solutions = solve_batch(targets)
        seconds = seconds_for(targets)
        steps = np.asarray(solutions.stats.num_steps)
        milliseconds = 1000 * seconds
        print(
            f"{batch_size:6d} {milliseconds:9.2f} {milliseconds / batch_size:11.3f} "
            f"{steps.max():10d} {steps.min():10d}"
        )


if __name__ == "__main__":
    main()
