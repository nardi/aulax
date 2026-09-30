"""Select the S2MPJ problems for the benchmarks and check each one.

Run with `python -m benchmarks.s2mpj.select`. The classification string of each
problem file decides whether it qualifies. Problems whose size is a parameter
are instantiated with their default arguments, in a separate process with a time
limit, to learn their size. Every candidate then gets its derivatives checked
against finite differences. The result goes to `selection.json` next to this
file, which is committed, so the selection only changes on purpose.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..processes import run_in_processes
from .fetch import COMMIT, PROBLEM_DIRECTORY, is_fetched
from .problems import load_problem, objective_and_gradient, objective_value


SELECTION_PATH = Path(__file__).parent / "selection.json"
"""Where the selection is written."""

MAX_VARIABLES = 100
"""The largest number of variables a selected problem may have."""

MAX_CONSTRAINTS = 100
"""The largest number of general constraints a selected problem may have."""

DERIVATIVE_TOLERANCE = 1e-4
"""The largest relative difference between analytic and finite-difference
derivatives, relative to the largest derivative."""


@dataclass
class Candidate:
    """A problem file with what its classification string says about it."""

    name: str
    """The problem name, which is also the file and class name."""
    classification: str
    """The classification string, such as `C-COOR2-AY-4-2`."""
    variables: int | None
    """The number of variables, or `None` if the user can choose it."""
    constraints: int | None
    """The number of general constraints, or `None` if the user can choose it."""
    subset: str
    """`general`, `bounds` for problems with only bounds, or `irregular`."""


@dataclass
class SelectedProblem:
    """A problem that passed the checks."""

    name: str
    """The problem name."""
    classification: str
    """The classification string."""
    variables: int
    """The number of variables with default arguments."""
    constraints: int
    """The number of general constraints with default arguments."""
    subset: str
    """`general`, `bounds` or `irregular`."""


def classify(name: str) -> Candidate | str:
    """Return the candidate for a problem file, or the reason it does not qualify."""
    source = (PROBLEM_DIRECTORY / f"{name}.py").read_text(encoding="utf-8")
    match = re.search(r'classification\s*=\s*"([^"]+)"', source)
    if match is None:
        return "no classification string"
    classification = match.group(1)
    parts = classification.split("-")
    if len(parts) != 5 or len(parts[1]) < 4:
        return f"unexpected classification {classification}"
    codes = parts[1]
    if codes[0] != "C":
        return "has integer or binary variables"
    if codes[2] in ("U", "X"):
        return "has no constraints beyond fixed variables"
    if codes[3] == "I":
        subset = "irregular"
    elif codes[2] == "B":
        subset = "bounds"
    else:
        subset = "general"
    variables = None if parts[3] == "V" else int(parts[3])
    constraints = None if parts[4] == "V" else int(parts[4])
    if variables is not None and variables > MAX_VARIABLES:
        return "too many variables"
    if constraints is not None and constraints > MAX_CONSTRAINTS:
        return "too many constraints"
    return Candidate(name, classification, variables, constraints, subset)


def _central_differences(function, x: np.ndarray, size: int) -> np.ndarray:
    """Return the Jacobian of `function` at `x` by central differences."""
    columns = []
    for index in range(x.size):
        step = 1e-6 * max(1.0, abs(x[index]))
        forward, backward = x.copy(), x.copy()
        forward[index] += step
        backward[index] -= step
        columns.append((function(forward) - function(backward)) / (2 * step))
    return np.stack(columns, axis=1).reshape(size, x.size)


def check_problem(name: str) -> tuple[int, int]:
    """Instantiate `name`, check its derivatives, and return its size.

    Raises an error that names the failed check. This runs in a separate process,
    so that a problem that hangs or crashes can be stopped.
    """
    problem = load_problem(name)
    n = int(problem.n)
    m = int(getattr(problem, "m", 0))
    if n > MAX_VARIABLES or m > MAX_CONSTRAINTS:
        raise ValueError(f"size {n} variables and {m} constraints is too large")
    x0 = np.asarray(problem.x0, dtype=float).ravel()
    lower = np.asarray(getattr(problem, "xlower", np.full(n, -np.inf)), float).ravel()
    upper = np.asarray(getattr(problem, "xupper", np.full(n, np.inf)), float).ravel()
    rng = np.random.default_rng(0)
    shifted = np.clip(x0 + 0.1 * (1.0 + np.abs(x0)) * rng.normal(size=n), lower, upper)

    def objective(x):
        return np.array([objective_value(problem, x.reshape(n, 1))])

    def constraints(x):
        return np.asarray(problem.cx(x.reshape(n, 1)), dtype=float).ravel()

    for point in (x0, shifted):
        _, gradient = objective_and_gradient(problem, point.reshape(n, 1))
        gradient = gradient.reshape(1, n)
        checks = [("gradient", gradient, _central_differences(objective, point, 1))]
        if m > 0:
            values, jacobian = problem.cJx(point.reshape(n, 1))
            if hasattr(jacobian, "toarray"):
                jacobian = jacobian.toarray()
            jacobian = np.asarray(jacobian, dtype=float).reshape(m, n)
            if not np.all(np.isfinite(values)):
                raise ValueError("constraint values are not finite")
            checks.append(
                ("Jacobian", jacobian, _central_differences(constraints, point, m))
            )
        for label, analytic, numeric in checks:
            if not (np.all(np.isfinite(analytic)) and np.all(np.isfinite(numeric))):
                raise ValueError(f"{label} is not finite")
            scale = max(1.0, float(np.max(np.abs(analytic), initial=0.0)))
            difference = float(np.max(np.abs(analytic - numeric), initial=0.0))
            if difference > DERIVATIVE_TOLERANCE * scale:
                raise ValueError(
                    f"{label} differs from finite differences by {difference:.1e}"
                )
    return n, m


def main() -> None:
    """Classify, instantiate and check every problem, and write the selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    arguments = parser.parse_args()
    if not is_fetched():
        raise SystemExit("Run `python -m benchmarks.s2mpj.fetch` first.")

    names = sorted(path.stem for path in PROBLEM_DIRECTORY.glob("*.py"))
    candidates: list[Candidate] = []
    excluded: dict[str, str] = {}
    for name in names:
        candidate = classify(name)
        if isinstance(candidate, str):
            excluded[name] = candidate
        else:
            candidates.append(candidate)
    print(f"{len(candidates)} of {len(names)} problems qualify by classification.")

    outcomes = run_in_processes(
        check_problem,
        [(candidate.name,) for candidate in candidates],
        timeout=arguments.timeout,
        workers=arguments.workers,
    )
    selected = []
    for candidate, outcome in zip(candidates, outcomes, strict=True):
        if outcome.result is None:
            excluded[candidate.name] = outcome.error or "unknown error"
            continue
        variables, constraints = outcome.result
        selected.append(
            SelectedProblem(
                candidate.name,
                candidate.classification,
                variables,
                constraints,
                candidate.subset,
            )
        )
    content = {
        "s2mpj_commit": COMMIT,
        "selected": [asdict(problem) for problem in selected],
        "excluded": dict(sorted(excluded.items())),
    }
    SELECTION_PATH.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    counts = {
        subset: sum(problem.subset == subset for problem in selected)
        for subset in ("general", "bounds", "irregular")
    }
    print(f"Selected {len(selected)} problems: {counts}. Wrote {SELECTION_PATH}.")


if __name__ == "__main__":
    main()
