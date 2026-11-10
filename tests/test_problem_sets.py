"""Tests for the problem sets that the benchmark scripts load by name."""

import numpy as np

from benchmarks.problem_sets import PROBLEM_SETS
from benchmarks.problems import BenchmarkProblem


def test_rescaled_set_has_the_problems_of_the_hock_schittkowski_set():
    assert (
        PROBLEM_SETS["rescaled"].problem_names()
        == PROBLEM_SETS["hock_schittkowski"].problem_names()
    )


def test_rescaled_problem_loads_with_the_name_it_is_requested_by():
    problem = PROBLEM_SETS["rescaled"].load_problem("hs71")
    assert isinstance(problem, BenchmarkProblem)
    assert problem.name == "hs71"


def test_rescaled_problem_changes_the_starting_point_and_objective():
    original = PROBLEM_SETS["hock_schittkowski"].load_problem("hs71")
    rescaled = PROBLEM_SETS["rescaled"].load_problem("hs71")
    assert np.shape(rescaled.y0) == np.shape(original.y0)
    assert not np.allclose(rescaled.y0, original.y0)
    assert rescaled.objective != original.objective
