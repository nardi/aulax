import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from problems import PROBLEMS, BenchmarkProblem, hs28, hs35

import aulax as alx


SOLVER = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-10, atol=1e-10))


@pytest.mark.parametrize("test_problem", PROBLEMS, ids=lambda problem: problem.name)
def test_hock_schittkowski(test_problem: BenchmarkProblem):
    assert test_problem.solution is not None
    assert test_problem.objective is not None
    solution = alx.minimise(test_problem.problem, SOLVER, jnp.asarray(test_problem.y0))
    np.testing.assert_allclose(solution.value, test_problem.solution, atol=1e-5)
    objective = test_problem.problem(solution.value, None, None).objective
    np.testing.assert_allclose(objective, test_problem.objective, atol=1e-5)
    assert solution.stats.infeasibility <= 1e-6
    assert solution.stats.stationarity <= 1e-4


def test_solution_multipliers_match_known_values():
    # The multipliers of HS71 from the Hock-Schittkowski collection, with the
    # sign of the equality multiplier following h = sum(y**2) - 40.
    test_problem = PROBLEMS[-1]
    solution = alx.minimise(test_problem.problem, SOLVER, jnp.asarray(test_problem.y0))
    np.testing.assert_allclose(solution.multipliers.equality, 0.16146857, atol=1e-5)
    np.testing.assert_allclose(
        solution.multipliers.inequality[:2], [0.55229366, 1.08787121], atol=1e-5
    )


def test_minimise_under_jit_and_vmap():
    def shifted_hs28(y, args, state):
        evaluation = hs28(y, None, None)
        return alx.Evaluation(
            objective=evaluation.objective, equality=evaluation.equality - args
        )

    @jax.jit
    @jax.vmap
    def solve(shift):
        return alx.minimise(shifted_hs28, SOLVER, jnp.zeros(3), shift).value

    shifts = jnp.array([0.0, 1.0, -2.0])
    batched = solve(shifts)
    for shift, value in zip(shifts, batched, strict=True):
        single = alx.minimise(shifted_hs28, SOLVER, jnp.zeros(3), shift).value
        np.testing.assert_allclose(value, single, atol=1e-6)
    # The residual is linear in the shift, so the minimizer is too.
    np.testing.assert_allclose(batched[0], [0.5, -0.5, 0.5], atol=1e-6)


def test_problem_runs_once_per_evaluation():
    shared_calls = 0

    def count_shared_computation(_):
        nonlocal shared_calls
        shared_calls += 1

    def counted_hs35(y, args, state):
        jax.debug.callback(count_shared_computation, y)
        return hs35(y, args, state)

    solution = alx.minimise(counted_hs35, SOLVER, jnp.array([0.5, 0.5, 0.5]))
    # One evaluation per loop iteration, plus one at the start to set up the
    # penalty and one at the end for the stationarity residual.
    assert shared_calls == int(solution.stats.num_steps) + 2


def test_state_is_threaded_between_evaluations():
    def counting_hs28(y, args, state):
        evaluation = hs28(y, args, None)
        return alx.Evaluation(
            objective=evaluation.objective,
            equality=evaluation.equality,
            state=state + 1,
            aux=state,
        )

    solution = alx.minimise(
        counting_hs28, SOLVER, jnp.array([-4.0, 1.0, 1.0]), state=jnp.array(0)
    )
    np.testing.assert_allclose(solution.value, [0.5, -0.5, 0.5], atol=1e-6)
    # The state is threaded from each accepted point, so the counter grows with
    # the number of accepted steps. The aux output holds the state that went into
    # the evaluation at the solution.
    assert solution.state == solution.aux + 1
    assert solution.state > solution.stats.num_outer_steps


def test_pytree_variables_and_constraints():
    def split_hs35(y, args, state):
        return hs35(jnp.append(y["pair"], y["single"]), args, state)

    y0 = {"pair": jnp.array([0.5, 0.5]), "single": jnp.array(0.5)}
    solution = alx.minimise(split_hs35, SOLVER, y0)
    np.testing.assert_allclose(solution.value["pair"], [4 / 3, 7 / 9], atol=1e-5)
    np.testing.assert_allclose(solution.value["single"], 4 / 9, atol=1e-5)
    assert set(solution.multipliers.inequality) == {"sum", "bounds"}


def test_infeasible_problem_is_reported():
    def infeasible(y, args, state):
        return alx.Evaluation(
            objective=jnp.sum(y**2), inequality=jnp.array([y[0] + 1.0, 1.0 - y[0]])
        )

    # A lower maximum penalty keeps the test short, since the inner solves get
    # slower as the penalty grows.
    solver = alx.AugmentedLagrangian(
        inner=optx.BFGS(rtol=1e-10, atol=1e-10),
        penalty_update=alx.AlgencanPenaltyUpdate(maximum_value=1e6),
    )
    solution = alx.minimise(infeasible, solver, jnp.array([0.0]), throw=False)
    assert solution.result == alx.RESULTS.infeasible
    np.testing.assert_allclose(solution.stats.penalty.inequality, [1e6, 1e6])


def test_failure_raises_when_throw_is_set():
    def infeasible(y, args, state):
        return alx.Evaluation(objective=y[0] ** 2, equality=y[0] ** 2 + 1.0)

    solver = alx.AugmentedLagrangian(
        inner=optx.BFGS(rtol=1e-10, atol=1e-10),
        penalty_update=alx.AlgencanPenaltyUpdate(maximum_value=1e6),
    )
    with pytest.raises(Exception, match="infeasible"):
        alx.minimise(infeasible, solver, jnp.array([1.0]))


def test_float32_solve():
    solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-6, atol=1e-6))
    solution = alx.minimise(hs28, solver, jnp.array([-4.0, 1.0, 1.0], jnp.float32))
    assert solution.value.dtype == jnp.float32
    assert solution.multipliers.equality.dtype == jnp.float32
    np.testing.assert_allclose(solution.value, [0.5, -0.5, 0.5], atol=1e-3)
