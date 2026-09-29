import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.metrics import classify_outcome, constraint_violation
from benchmarks.problems import PROBLEM_SETS, BenchmarkProblem
from benchmarks.records import (
    SolveRecord,
    new_result_file,
    read_result_file,
    write_result_file,
)


ALL_PROBLEMS = [
    pytest.param(problem, id=f"{set_name}-{problem.name}")
    for set_name, problem_set in PROBLEM_SETS.items()
    for problem in problem_set()
]


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
def test_problem_evaluates_to_finite_values(problem: BenchmarkProblem):
    evaluation = problem.problem(
        jax.tree.map(jnp.asarray, problem.y0), problem.args, None
    )
    assert jnp.shape(evaluation.objective) == ()
    for leaf in jax.tree.leaves(evaluation):
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize("problem", ALL_PROBLEMS)
def test_known_solution_matches_known_objective(problem: BenchmarkProblem):
    if problem.solution is None or problem.objective is None:
        pytest.skip("The problem has no known solution.")
    evaluation = problem.problem(
        jax.tree.map(jnp.asarray, problem.solution), problem.args, None
    )
    # Published solutions are rounded to about seven digits, so both checks use
    # a tolerance that allows for that. Constraints with coefficients of 1000, as
    # in HS74, turn that rounding into violations of a few times 1e-4.
    np.testing.assert_allclose(
        evaluation.objective, problem.objective, rtol=1e-5, atol=1e-6
    )
    assert constraint_violation(evaluation) < 1e-3


def test_outcome_classification():
    assert classify_outcome(True, 0.0, 1.0, 1.0) == "solved"
    assert classify_outcome(True, 0.0, 1.0, None) == "solved"
    assert classify_outcome(True, 0.0, 0.5, 1.0) == "below_reference"
    assert classify_outcome(True, 0.0, 2.0, 1.0) == "other_minimum"
    assert classify_outcome(True, 1e-3, 1.0, 1.0) == "failed"
    assert classify_outcome(False, 0.0, 1.0, 1.0) == "failed"


def test_result_file_round_trip(tmp_path):
    record = SolveRecord(
        problem="hs6",
        variables=2,
        equality_constraints=1,
        inequality_constraints=0,
        outcome="solved",
        message="successful",
        evaluations=10,
    )
    result_file = new_result_file("baseline", "hock_schittkowski", [record])
    path = tmp_path / "result.json"
    write_result_file(result_file, path)
    assert read_result_file(path) == result_file
