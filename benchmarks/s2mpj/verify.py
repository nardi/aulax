"""Check handwritten benchmark problems against their S2MPJ versions.

Run with `python -m benchmarks.s2mpj.verify`. For every Hock-Schittkowski problem
in the benchmark suite, this compares the starting point, the known objective,
the objective at random points, and the constraints. Constraints may be written
in a different order or scaled by a positive factor, so each handwritten residual
must be proportional to exactly one S2MPJ residual of the same kind.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..metrics import flat_leaves
from ..problems import BenchmarkProblem
from .fetch import is_fetched
from .problems import ConstraintMapping, evaluate_numpy, known_objective, load_problem


WRONG_KNOWN_OBJECTIVES = frozenset({"HS14", "HS33", "HS44", "HS72", "HS75", "HS81"})
"""S2MPJ problems whose `LO SOLTN` comment disagrees with the Hock-Schittkowski book.

For each of these, the book's minimizer is feasible and reaches the book's
objective, which the benchmark smoke tests check. For HS14, HS33 and HS44 that
objective is lower than S2MPJ's. HS75 carries the value of HS74, and HS81 is off
by a factor of ten. The comparison skips the known objective for these.
"""


def _unmatched_residuals(
    ours: np.ndarray, theirs: np.ndarray, allow_negative: bool
) -> list[int]:
    """Return the columns of `ours` that match no column of `theirs`.

    Both arrays hold one residual per column and one random point per row. Two
    columns match when one is a multiple of the other. For inequalities the
    multiple has to be positive, since a negative one flips the constraint.
    Each column of `theirs` matches at most one column of `ours`.
    """
    used: set[int] = set()
    unmatched = []
    for i in range(ours.shape[1]):
        column = ours[:, i]
        found = False
        for j in range(theirs.shape[1]):
            if j in used:
                continue
            other = theirs[:, j]
            norms = np.linalg.norm(column) * np.linalg.norm(other)
            if norms == 0.0:
                continue
            cosine = float(column @ other) / norms
            if cosine > 1 - 1e-9 or (allow_negative and cosine < -1 + 1e-9):
                used.add(j)
                found = True
                break
        if not found:
            unmatched.append(i)
    return unmatched


def compare_with_s2mpj(
    problem: BenchmarkProblem, s2mpj_name: str, points: int = 20, seed: int = 0
) -> list[str]:
    """Return a list of differences between `problem` and its S2MPJ version."""
    s2mpj_problem = load_problem(s2mpj_name)
    mapping = ConstraintMapping.from_problem(s2mpj_problem)
    differences = []

    x0 = np.asarray(s2mpj_problem.x0, dtype=float).ravel()
    if not np.allclose(x0, problem.y0):
        differences.append(f"starting point {problem.y0} differs from {x0}")
    reference = known_objective(s2mpj_name)
    if (
        reference is not None
        and s2mpj_name not in WRONG_KNOWN_OBJECTIVES
        and problem.objective is not None
        and not np.isclose(problem.objective, reference, rtol=1e-5, atol=1e-6)
    ):
        differences.append(
            f"known objective {problem.objective} differs from {reference}"
        )

    rng = np.random.default_rng(seed)
    scale = np.maximum(1.0, np.abs(problem.y0))
    ours = {"objective": [], "equality": [], "inequality": []}
    theirs = {"objective": [], "equality": [], "inequality": []}
    for _ in range(points):
        x = problem.y0 + scale * rng.normal(size=problem.y0.shape)
        evaluation = problem.problem(jnp.asarray(x), problem.args, None)
        objective, equality, inequality = evaluate_numpy(s2mpj_problem, mapping, x)
        ours["objective"].append(float(evaluation.objective))
        ours["equality"].append(flat_leaves(evaluation.equality))
        ours["inequality"].append(flat_leaves(evaluation.inequality))
        theirs["objective"].append(objective)
        theirs["equality"].append(flat_leaves(equality))
        theirs["inequality"].append(flat_leaves(inequality))

    if not np.allclose(ours["objective"], theirs["objective"], rtol=1e-9, atol=1e-9):
        differences.append("objective values differ at random points")
    for kind in ("equality", "inequality"):
        our_values = np.array(ours[kind])
        their_values = np.array(theirs[kind])
        if our_values.shape[1] != their_values.shape[1]:
            differences.append(
                f"{our_values.shape[1]} {kind} residuals, S2MPJ has "
                f"{their_values.shape[1]}"
            )
        unmatched = _unmatched_residuals(
            our_values, their_values, allow_negative=kind == "equality"
        )
        if unmatched:
            differences.append(f"{kind} residuals {unmatched} match nothing in S2MPJ")
    return differences


def main() -> None:
    """Compare every Hock-Schittkowski benchmark problem and print the result."""
    from ..problems.hock_schittkowski import PROBLEMS

    if not is_fetched():
        raise SystemExit("Run `python -m benchmarks.s2mpj.fetch` first.")
    jax.config.update("jax_enable_x64", True)
    failures = 0
    for problem in PROBLEMS:
        differences = compare_with_s2mpj(problem, problem.name.upper())
        status = "ok" if not differences else "; ".join(differences)
        failures += bool(differences)
        print(f"{problem.name:8} {status}")
    print(f"{failures} of {len(PROBLEMS)} problems differ.")


if __name__ == "__main__":
    main()
