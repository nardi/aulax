import importlib.util
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from problems import PROBLEMS

import aulax as alx
from aulax._hessian import (
    Indices,
    augmented_lagrangian_pattern,
    shifted_trust_region_step,
    trust_region_step,
)


requires_sparse_libraries = pytest.mark.skipif(
    importlib.util.find_spec("asdex") is None
    or importlib.util.find_spec("splineax") is None,
    reason="Install the `sparse` dependency group.",
)


def banded(y, args):
    """Return a function with a tridiagonal Hessian, as the inner minimiser sees it."""
    value = (
        jnp.sum((y[1:] - y[:-1]) ** 2) + 0.01 * jnp.sum(y**4) + jnp.sum((y - args) ** 2)
    )
    return value, None


def prepare(strategy, size=40):
    y = jnp.linspace(-1.0, 1.0, size)
    args = jnp.sin(y)
    return strategy.prepare(banded, y, args), y, args


def scalar_of(y, args):
    return lambda flat: banded(flat, args)[0]


def test_matrix_free_product_matches_the_dense_hessian():
    dense, y, args = prepare(alx.DenseHessian())
    free, _, _ = prepare(alx.MatrixFreeHessian())
    scalar = scalar_of(y, args)
    data = dense.evaluate(scalar, y)
    vector = jnp.cos(3.0 * y)
    expected = data @ vector
    actual = free.matvec(scalar, free.evaluate(scalar, y), vector)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_shifted_step_matches_the_eigendecomposition_step_for_a_definite_matrix():
    rng = np.random.default_rng(0)
    root = rng.normal(size=(6, 6))
    matrix = jnp.asarray(root @ root.T + 6.0 * np.eye(6))
    gradient = jnp.asarray(rng.normal(size=6))
    identity = jnp.eye(6)

    def solve(shift):
        factor = jnp.linalg.cholesky(matrix + shift * identity)
        positive = jnp.all(jnp.isfinite(factor))
        step = -jax.scipy.linalg.cho_solve((factor, True), gradient)
        correction = jax.scipy.linalg.cho_solve((factor, True), step)
        return step, correction, positive

    for radius in (10.0, 0.5, 0.05):
        radius = jnp.asarray(radius)
        expected = trust_region_step(matrix, gradient, radius)
        actual = shifted_trust_region_step(solve, gradient, radius, 10.0)
        np.testing.assert_allclose(
            jnp.linalg.norm(actual), jnp.linalg.norm(expected), rtol=2e-2
        )


def test_shifted_trust_region_step_stays_in_the_region_for_an_indefinite_matrix():
    matrix = jnp.diag(jnp.array([2.0, -3.0, 1.0]))
    gradient = jnp.array([1.0, 0.5, -0.2])
    identity = jnp.eye(3)

    def solve(shift):
        factor = jnp.linalg.cholesky(matrix + shift * identity)
        positive = jnp.all(jnp.isfinite(factor))
        step = -jax.scipy.linalg.cho_solve((factor, True), gradient)
        correction = jax.scipy.linalg.cho_solve((factor, True), step)
        return step, correction, positive

    step = shifted_trust_region_step(solve, gradient, jnp.asarray(0.5), 3.0)
    assert jnp.linalg.norm(step) <= 0.5 * 1.02
    assert gradient @ step + 0.5 * step @ matrix @ step < 0.0


def test_pattern_of_the_augmented_lagrangian_is_the_union_with_the_product():
    # The objective couples neighbours, and one constraint involves variables 0 and 3.
    objective = Indices([0, 1, 1, 2], [0, 1, 2, 2])
    jacobian = Indices([0, 0], [0, 3])
    pattern = augmented_lagrangian_pattern(
        alx.ProblemSparsity(objective, jacobian), size=4
    )
    assert pattern is not None
    entries = set(zip(pattern.rows.tolist(), pattern.cols.tolist(), strict=True))
    assert entries == {
        (0, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (0, 3),
        (3, 0),
        (3, 3),
    }


def test_pattern_is_unknown_when_a_piece_is_missing():
    assert (
        augmented_lagrangian_pattern(
            alx.ProblemSparsity(objective_hessian=Indices([0], [0])), size=3
        )
        is None
    )


def test_sparsity_accepts_several_forms():
    dense = np.eye(3)
    forms = [
        dense,
        (np.arange(3), np.arange(3)),
        BCOO.fromdense(jnp.eye(3)),
    ]
    for form in forms:
        sparsity = alx.ProblemSparsity(objective_hessian=form, constraint_jacobian=form)
        assert sparsity.objective_hessian is not None
        assert sorted(sparsity.objective_hessian.rows.tolist()) == [0, 1, 2]


def test_sparsity_with_equal_content_compares_equal():
    first = alx.ProblemSparsity(np.eye(3), np.eye(3))
    second = alx.ProblemSparsity(np.eye(3), np.eye(3))
    assert first == second
    assert hash(first) == hash(second)


def test_jacobi_preconditioner_inverts_the_diagonal():
    matrix = jnp.diag(jnp.array([1.0, 4.0, 10.0])) + 0.1 * jnp.ones((3, 3))
    point = jnp.zeros(3)

    def product(vector):
        return matrix @ vector

    exact = alx.JacobiPreconditioner().build(product, point)
    np.testing.assert_allclose(exact(jnp.ones(3)), 1.0 / jnp.diag(matrix), rtol=1e-12)
    probed = alx.JacobiPreconditioner(probes=2000).build(product, point)
    np.testing.assert_allclose(probed(jnp.ones(3)), 1.0 / jnp.diag(matrix), rtol=0.1)


def test_a_plain_function_can_serve_as_a_preconditioner():
    calls = []

    def build(product, point):
        calls.append(True)
        return lambda vector: 0.5 * vector

    free, y, args = prepare(alx.MatrixFreeHessian(preconditioner=build), size=10)
    scalar = scalar_of(y, args)
    gradient = jax.grad(scalar)(y)
    step = free.step(scalar, free.evaluate(scalar, y), gradient, jnp.asarray(1.0))
    assert calls
    assert jnp.linalg.norm(step) <= 1.0 + 1e-8


def test_auto_chooses_dense_for_small_problems():
    auto, _, _ = prepare(alx.AutoHessian(dense_limit=100), size=40)
    assert auto.mode == "dense"


@requires_sparse_libraries
def test_auto_chooses_sparse_for_a_large_banded_problem():
    auto, _, _ = prepare(alx.AutoHessian(dense_limit=10), size=200)
    assert auto.mode == "sparse"


def test_auto_falls_back_to_matrix_free_with_a_warning(monkeypatch):
    def fail(*arguments, **keywords):
        raise RuntimeError("detection failed")

    monkeypatch.setattr(alx.SparseHessian, "prepare", fail)
    with pytest.warns(UserWarning, match="detection failed"):
        auto, _, _ = prepare(alx.AutoHessian(dense_limit=10), size=40)
    assert auto.mode == "matrix_free"


@pytest.mark.parametrize(
    "hessian",
    [
        alx.DenseHessian(),
        alx.DenseHessian(eigendecomposition_limit=0),
        alx.MatrixFreeHessian(),
        alx.MatrixFreeHessian(preconditioner=alx.JacobiPreconditioner()),
        alx.AutoHessian(),
    ],
    ids=["dense", "dense_cholesky", "matrix_free", "jacobi", "auto"],
)
def test_every_mode_solves_a_constrained_problem(hessian):
    test_problem = PROBLEMS[-1]
    assert test_problem.solution is not None
    solver = alx.AugmentedLagrangian(inner=alx.Newton(hessian=hessian))
    solution = alx.minimise(
        test_problem.problem, solver, jnp.asarray(test_problem.y0), max_steps=5000
    )
    np.testing.assert_allclose(solution.value, test_problem.solution, atol=1e-4)


def pentadiagonal_problem(y, args, state):
    """A banded problem with known sparsity, for tests of user-given patterns."""
    return alx.Evaluation(
        objective=jnp.sum((y - 0.3) ** 2) + 0.1 * jnp.sum(jnp.diff(y) ** 2),
        equality=y[1:] - 0.5 * y[:-1] - 0.1 * jnp.sin(y[1:]),
        inequality=jnp.concatenate([-y - 1.0, y - 2.0]),
    )


def banded_sparsity(size):
    hessian = np.eye(size) + np.eye(size, k=1) + np.eye(size, k=-1)
    equality = np.zeros((size - 1, size))
    for row in range(size - 1):
        equality[row, row : row + 2] = 1.0
    bounds = np.concatenate([np.eye(size), np.eye(size)], axis=0)
    return alx.ProblemSparsity(
        objective_hessian=hessian,
        constraint_jacobian=np.concatenate([equality, bounds], axis=0),
    )


@requires_sparse_libraries
def test_a_given_pattern_gives_the_same_solution_as_detection():
    size = 30
    y0 = jnp.zeros(size)
    given = alx.AugmentedLagrangian(
        inner=alx.Newton(
            hessian=alx.SparseHessian(
                sparsity=banded_sparsity(size), check_pattern=True
            )
        )
    )
    detected = alx.AugmentedLagrangian(inner=alx.Newton(hessian=alx.SparseHessian()))
    first = alx.minimise(pentadiagonal_problem, given, y0, max_steps=3000)
    second = alx.minimise(pentadiagonal_problem, detected, y0, max_steps=3000)
    np.testing.assert_allclose(first.value, second.value, atol=1e-5)


@requires_sparse_libraries
def test_a_wrong_pattern_is_reported_by_the_check():
    size = 30
    # The objective Hessian is diagonal here, but the problem couples neighbours.
    wrong = alx.ProblemSparsity(
        objective_hessian=np.eye(size),
        constraint_jacobian=np.eye(size)[: size - 1],
    )
    solver = alx.AugmentedLagrangian(
        inner=alx.Newton(hessian=alx.SparseHessian(sparsity=wrong, check_pattern=True))
    )
    with pytest.raises(Exception, match="sparsity pattern"):
        alx.minimise(pentadiagonal_problem, solver, jnp.zeros(size), max_steps=500)


def test_no_warning_for_a_plain_default_solve():
    test_problem = PROBLEMS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alx.minimise(
            test_problem.problem,
            alx.AugmentedLagrangian(),
            jnp.asarray(test_problem.y0),
        )
