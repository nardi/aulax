"""Time batches of small problems solved under `jax.vmap`.

Example: `python -m benchmarks.batched --config baseline`. The problems in a
batch differ in their `args`, so they have different solutions and take
different numbers of steps. The batch runs until its slowest member finishes.
"""

import argparse
import dataclasses
import datetime
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import aulax as alx

from .configs import CONFIGURATIONS, AulaxConfiguration
from .metrics import FEASIBILITY_TOLERANCE
from .problems.batched import FAMILIES, BatchedFamily
from .records import RESULTS_DIRECTORY, current_commit, package_versions


BATCH_SIZES = (1, 10, 100, 1000)
"""The batch sizes each family runs at."""


@dataclass
class BatchRecord:
    """The cost of solving one batch of one family."""

    family: str
    """The name of the problem family."""
    batch_size: int
    """The number of problems in the batch."""
    solved: int
    """How many problems converged to a feasible point."""
    seconds: float
    """The fastest time of the timed runs of the whole batch."""
    compile_seconds: float
    """The time of the first run, which includes compilation."""
    minimum_steps: int
    """The fewest loop iterations any problem in the batch needed."""
    median_steps: float
    """The median loop iterations over the batch."""
    maximum_steps: int
    """The most loop iterations any problem in the batch needed."""


def run_family(
    family: BatchedFamily,
    configuration: AulaxConfiguration,
    batch_size: int,
    repeats: int,
    seed: int = 0,
) -> BatchRecord:
    """Solve one batch of `family` under `jax.vmap` and return its record."""
    args = jnp.asarray(family.sample_args(np.random.default_rng(seed), batch_size))
    y0 = jax.tree.map(jnp.asarray, family.y0)

    @jax.jit
    def solve_batch(batch_args):
        return jax.vmap(
            lambda member_args: alx.minimise(
                family.problem,
                configuration.solver,
                y0,
                member_args,
                max_steps=configuration.max_steps,
                throw=False,
            )
        )(batch_args)

    start = time.perf_counter()
    solutions = jax.block_until_ready(solve_batch(args))
    compile_seconds = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(solve_batch(args))
        times.append(time.perf_counter() - start)

    steps = np.asarray(solutions.stats.num_steps)
    converged = np.asarray(solutions.result == alx.RESULTS.successful)
    feasible = np.asarray(solutions.stats.infeasibility) <= FEASIBILITY_TOLERANCE
    return BatchRecord(
        family=family.name,
        batch_size=batch_size,
        solved=int(np.sum(converged & feasible)),
        seconds=min(times),
        compile_seconds=compile_seconds,
        minimum_steps=int(steps.min()),
        median_steps=float(np.median(steps)),
        maximum_steps=int(steps.max()),
    )


def main() -> None:
    """Parse the command line, run the batches and write a result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    aulax_configurations = sorted(
        name
        for name, configuration in CONFIGURATIONS.items()
        if isinstance(configuration, AulaxConfiguration)
    )
    parser.add_argument("--config", required=True, choices=aulax_configurations)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIRECTORY)
    arguments = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    configuration = CONFIGURATIONS[arguments.config]
    assert isinstance(configuration, AulaxConfiguration)
    records = []
    print(f"device: {jax.devices()[0].device_kind}")
    for family in FAMILIES:
        for batch_size in BATCH_SIZES:
            record = run_family(family, configuration, batch_size, arguments.repeats)
            records.append(record)
            print(
                f"{family.name:24} {batch_size:6d} solved {record.solved:5d} "
                f"{1000 * record.seconds:10.2f} ms "
                f"{1000 * record.seconds / batch_size:8.3f} ms/problem "
                f"steps {record.minimum_steps}-{record.maximum_steps}"
            )
    content = {
        "configuration": arguments.config,
        "commit": current_commit(),
        "versions": package_versions(),
        "device": jax.devices()[0].device_kind,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "records": [dataclasses.asdict(record) for record in records],
    }
    path = arguments.output_dir / f"batched-{arguments.config}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
