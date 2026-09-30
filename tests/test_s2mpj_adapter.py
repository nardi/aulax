import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

import aulax as alx
from benchmarks.metrics import flat_leaves
from benchmarks.problems.hock_schittkowski import hs71
from benchmarks.s2mpj.adapter import adapted_problem
from benchmarks.s2mpj.fetch import is_fetched


pytestmark = pytest.mark.skipif(
    not is_fetched(), reason="Run `python -m benchmarks.s2mpj.fetch` first."
)


def test_adapted_hs71_matches_handwritten_values():
    adapted = adapted_problem("HS71")
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = jnp.asarray(1.0 + 4.0 * rng.random(4))
        ours = hs71(y, None, None)
        theirs = adapted.problem(y, None, None)
        np.testing.assert_allclose(theirs.objective, ours.objective)
        # Both have one equality constraint, and the same number of inequalities.
        assert flat_leaves(theirs.equality).size == 1
        assert flat_leaves(theirs.inequality).size == flat_leaves(ours.inequality).size


@pytest.mark.parametrize("name", ["HS71", "HS21", "HS80"])
def test_adapted_derivatives_match_finite_differences(name):
    adapted = adapted_problem(name)

    def flat_outputs(y):
        evaluation = adapted.problem(y, None, None)
        leaves = [evaluation.objective, *jax.tree.leaves(evaluation.equality)]
        leaves += jax.tree.leaves(evaluation.inequality)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    y = jnp.asarray(adapted.y0) + 0.1
    jacobian = jax.jacfwd(flat_outputs)(y)
    reverse = jax.jacrev(flat_outputs)(y)
    step = 1e-6
    finite_differences = np.stack(
        [
            (flat_outputs(y.at[i].add(step)) - flat_outputs(y.at[i].add(-step)))
            / (2 * step)
            for i in range(y.size)
        ],
        axis=1,
    )
    np.testing.assert_allclose(jacobian, finite_differences, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(reverse, jacobian, rtol=1e-12, atol=1e-12)


def test_aulax_solves_adapted_hs71():
    adapted = adapted_problem("HS71")
    solver = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
    solution = alx.minimise(adapted.problem, solver, jnp.asarray(adapted.y0))
    np.testing.assert_allclose(
        solution.value, [1.0, 4.7429994, 3.8211503, 1.3794082], atol=1e-4
    )


@pytest.mark.parametrize("name", ["HS71", "HS21", "HS80", "AIRCRFTA"])
def test_lagrangian_hessian_matches_finite_differences(name):
    adapted = adapted_problem(name)
    assert adapted.lagrangian_hessian is not None
    y = jnp.asarray(adapted.y0) + 0.1
    evaluation = adapted.problem(y, None, None)
    residual_count = (
        flat_leaves(evaluation.equality).size + flat_leaves(evaluation.inequality).size
    )
    rng = np.random.default_rng(1)
    multipliers = rng.normal(size=residual_count)
    objective_factor = 0.7

    def weighted_lagrangian(point):
        evaluation = adapted.problem(point, None, None)
        leaves = jax.tree.leaves(evaluation.equality)
        leaves += jax.tree.leaves(evaluation.inequality)
        residuals = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        return objective_factor * evaluation.objective + multipliers @ residuals

    gradient = jax.grad(weighted_lagrangian)
    step = 1e-6
    finite_differences = np.stack(
        [
            (gradient(y.at[i].add(step)) - gradient(y.at[i].add(-step))) / (2 * step)
            for i in range(y.size)
        ],
        axis=1,
    )
    hessian = adapted.lagrangian_hessian(np.asarray(y), objective_factor, multipliers)
    np.testing.assert_allclose(hessian, finite_differences, rtol=1e-4, atol=1e-5)
