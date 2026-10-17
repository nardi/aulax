import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jaxtyping import Array, Float, Scalar
from problems import PROBLEMS, BenchmarkProblem

import aulax as alx
from aulax._newton import trust_region_step


def rosenbrock(y: Float[Array, " 2"], args: None) -> tuple[Scalar, None]:
    """Return the Rosenbrock function, which has its minimum at (1, 1)."""
    return (1.0 - y[0]) ** 2 + 100.0 * (y[1] - y[0] ** 2) ** 2, None


def quadratic_model(hessian, gradient, step):
    """Return the value of the quadratic model at `step`."""
    return gradient @ step + 0.5 * step @ hessian @ step


def test_trust_region_step_is_newton_step_inside_the_region():
    hessian = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    gradient = jnp.array([1.0, -2.0])
    step = trust_region_step(hessian, gradient, jnp.asarray(10.0))
    np.testing.assert_allclose(step, -jnp.linalg.solve(hessian, gradient), atol=1e-12)


def test_trust_region_step_reaches_the_boundary_when_newton_is_too_long():
    hessian = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    gradient = jnp.array([1.0, -2.0])
    radius = 0.1
    step = trust_region_step(hessian, gradient, jnp.asarray(radius))
    np.testing.assert_allclose(jnp.linalg.norm(step), radius, rtol=1e-10)


def test_trust_region_step_decreases_the_model_for_an_indefinite_hessian():
    hessian = jnp.array([[2.0, 0.0], [0.0, -3.0]])
    gradient = jnp.array([1.0, 0.5])
    radius = 1.5
    step = trust_region_step(hessian, gradient, jnp.asarray(radius))
    np.testing.assert_allclose(jnp.linalg.norm(step), radius, rtol=1e-8)
    assert quadratic_model(hessian, gradient, step) < 0.0
    # No point on the boundary does better than the step.
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 721)
    boundary = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    values = jax.vmap(lambda point: quadratic_model(hessian, gradient, point))(boundary)
    assert quadratic_model(hessian, gradient, step) <= values.min() + 1e-6


def test_trust_region_step_with_a_huge_curvature_ratio():
    # A penalty term makes the Hessian ill-conditioned, and the Newton step has
    # to stay accurate in the direction with the small curvature.
    hessian = jnp.diag(jnp.array([1e12, 1.0]))
    gradient = jnp.array([1.0, 1.0])
    step = trust_region_step(hessian, gradient, jnp.asarray(10.0))
    np.testing.assert_allclose(step, [-1e-12, -1.0], rtol=1e-8)


def test_minimises_rosenbrock_in_few_evaluations():
    solution = optx.minimise(
        rosenbrock,
        alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        jnp.array([-1.2, 1.0]),
        has_aux=True,
        max_steps=100,
    )
    np.testing.assert_allclose(solution.value, [1.0, 1.0], atol=1e-7)
    assert int(solution.stats["num_steps"]) < 40


def test_leaves_a_saddle_point_through_negative_curvature():
    def double_well(y, args):
        return jnp.sum(y**4 - 2.0 * y**2), None

    solution = optx.minimise(
        double_well,
        alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        jnp.array([1e-3, -1e-3]),
        has_aux=True,
        max_steps=100,
    )
    np.testing.assert_allclose(jnp.abs(solution.value), [1.0, 1.0], atol=1e-7)


def test_evaluates_the_function_once_per_step():
    evaluation_count = 0

    def count_evaluation(_):
        nonlocal evaluation_count
        evaluation_count += 1

    def counted_rosenbrock(y, args):
        jax.debug.callback(count_evaluation, y)
        return rosenbrock(y, args)

    solution = optx.minimise(
        counted_rosenbrock,
        alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        jnp.array([-1.2, 1.0]),
        has_aux=True,
        max_steps=100,
        throw=False,
    )
    jax.effects_barrier()
    assert evaluation_count == int(solution.stats["num_steps"])


def test_handles_pytree_variables():
    def fn(y, args):
        return (y["a"] - 1.0) ** 2 + jnp.sum((y["b"] + 2.0) ** 2), None

    solution = optx.minimise(
        fn,
        alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        {"a": jnp.array(0.0), "b": jnp.zeros(2)},
        has_aux=True,
    )
    np.testing.assert_allclose(solution.value["a"], 1.0, atol=1e-8)
    np.testing.assert_allclose(solution.value["b"], [-2.0, -2.0], atol=1e-8)


@pytest.mark.parametrize("test_problem", PROBLEMS, ids=lambda problem: problem.name)
def test_augmented_lagrangian_with_newton_inner_solves(test_problem: BenchmarkProblem):
    assert test_problem.solution is not None
    solver = alx.AugmentedLagrangian(
        inner=alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        stationarity_tolerance=1e-6,
    )
    solution = alx.minimise(test_problem.problem, solver, jnp.asarray(test_problem.y0))
    np.testing.assert_allclose(solution.value, test_problem.solution, atol=1e-5)
    assert solution.stats.infeasibility <= 1e-6
    assert solution.stats.stationarity <= 1e-5


def test_newton_inner_solves_need_fewer_steps_than_bfgs():
    test_problem = PROBLEMS[-1]
    bfgs = alx.AugmentedLagrangian(
        inner=optx.BFGS(rtol=1e-10, atol=1e-10), stationarity_tolerance=1e-6
    )
    newton = alx.AugmentedLagrangian(
        inner=alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        stationarity_tolerance=1e-6,
    )
    y0 = jnp.asarray(test_problem.y0)
    bfgs_steps = alx.minimise(test_problem.problem, bfgs, y0).stats.num_steps
    newton_steps = alx.minimise(test_problem.problem, newton, y0).stats.num_steps
    assert int(newton_steps) < int(bfgs_steps) // 2
