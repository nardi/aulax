"""Tests for the NLopt configurations of the benchmarks."""

import pytest

from benchmarks.configs import CONFIGURATIONS
from benchmarks.problem_sets import PROBLEM_SETS


pytest.importorskip("nlopt")


@pytest.mark.parametrize("configuration", ["nlopt_auglag_lbfgs", "nlopt_auglag_slsqp"])
def test_auglag_solves_hock_schittkowski_71(configuration: str):
    problem = PROBLEM_SETS["hock_schittkowski"].load_problem("hs71")
    record = CONFIGURATIONS[configuration].solve(problem, 0)
    assert record.outcome == "solved"
    assert record.infeasibility < 1e-6
    assert record.evaluations is not None and record.evaluations > 0
