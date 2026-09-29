"""Result records and the JSON files that hold them."""

import dataclasses
import datetime
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax


RESULTS_DIRECTORY = Path(__file__).parent / "results"
"""Where result files go unless the scripts are told otherwise."""


@dataclass
class SolveRecord:
    """The outcome and cost of one solve of one problem."""

    problem: str
    """The name of the problem."""
    variables: int
    """The number of variables."""
    equality_constraints: int
    """The number of equality residuals."""
    inequality_constraints: int
    """The number of inequality residuals, including bounds."""
    outcome: str
    """One of the outcomes defined in `benchmarks.metrics`."""
    message: str
    """The solver's own message, or the error for an `"error"` outcome."""
    evaluations: int | None = None
    """The number of calls of the problem function, with or without derivatives."""
    derivative_evaluations: int | None = None
    """How many of `evaluations` also computed derivatives."""
    outer_steps: int | None = None
    """The number of multiplier updates, for augmented Lagrangian solvers."""
    objective: float | None = None
    """The objective at the returned point."""
    known_objective: float | None = None
    """The known optimal objective, if there is one."""
    solution_error: float | None = None
    """The largest absolute difference from the known minimizer, if there is one."""
    infeasibility: float | None = None
    """The largest constraint violation at the returned point."""
    stationarity: float | None = None
    """The largest element of the gradient of the Lagrangian, if the solver reports
    multipliers."""
    seconds: float | None = None
    """The fastest time of the timed solves."""
    compile_seconds: float | None = None
    """The time of the first solve, which includes compilation."""


@dataclass
class ResultFile:
    """The records of one configuration on one problem set, with their context."""

    configuration: str
    """The name of the solver configuration."""
    problem_set: str
    """The name of the problem set."""
    commit: str
    """The aulax commit the results came from, with `-dirty` for local changes."""
    versions: dict[str, str]
    """The versions of the packages that affect the results."""
    device: str
    """The JAX device the solves ran on."""
    date: str
    """When the run finished, in ISO format."""
    records: list[SolveRecord]
    """One record per problem."""


def current_commit() -> str:
    """Return the short hash of the current commit, marked if the tree is dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def package_versions() -> dict[str, str]:
    """Return the versions of aulax and the packages it builds on."""
    names = ["aulax", "jax", "jaxlib", "equinox", "optimistix", "lineax", "scipy"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def new_result_file(
    configuration: str, problem_set: str, records: list[SolveRecord]
) -> ResultFile:
    """Return a result file with the current commit, versions, device and date."""
    return ResultFile(
        configuration=configuration,
        problem_set=problem_set,
        commit=current_commit(),
        versions=package_versions(),
        device=jax.devices()[0].device_kind,
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        records=records,
    )


def write_result_file(result_file: ResultFile, path: Path) -> None:
    """Write `result_file` to `path` as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result_file), indent=2) + "\n")


def read_result_file(path: Path) -> ResultFile:
    """Read a result file written by `write_result_file`."""
    content = json.loads(path.read_text())
    records = [SolveRecord(**record) for record in content.pop("records")]
    return ResultFile(**content, records=records)
