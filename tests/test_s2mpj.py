import pytest

from benchmarks.problems.hock_schittkowski import PROBLEMS
from benchmarks.s2mpj.fetch import is_fetched
from benchmarks.s2mpj.verify import compare_with_s2mpj


pytestmark = pytest.mark.skipif(
    not is_fetched(), reason="Run `python -m benchmarks.s2mpj.fetch` first."
)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda problem: problem.name)
def test_hock_schittkowski_problem_matches_s2mpj(problem):
    assert compare_with_s2mpj(problem, problem.name.upper()) == []
