import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

import aulax as alx


requires_sparse_libraries = pytest.mark.skipif(
    importlib.util.find_spec("asdex") is None
    or importlib.util.find_spec("splineax") is None,
    reason="Install the `sparse` dependency group.",
)

HESSIAN_MODES = [
    pytest.param(alx.DenseHessian(), id="dense"),
    pytest.param(alx.DenseHessian(eigendecomposition_limit=1), id="cholesky"),
    pytest.param(alx.MatrixFreeHessian(), id="matrix_free"),
    pytest.param(alx.SparseHessian(), id="sparse", marks=requires_sparse_libraries),
]


def rosenbrock(y, args):
    return (1.0 - y[0]) ** 2 + 100.0 * (y[1] - y[0] ** 2) ** 2, None


def quadratic_problem():
    """Return a convex quadratic with a known solution and two active bounds.

    The solution is at the lower bound of the first variable and at the upper bound
    of the third, and the gradient points out of the box at both. The second and
    fourth variables are free, so the gradient is zero there.
    """
    matrix = jnp.array(
        [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.5, 0.3],
            [0.5, 0.5, 2.0, 0.2],
            [0.0, 0.3, 0.2, 1.5],
        ]
    )
    solution = jnp.array([0.0, 0.7, 1.0, -0.3])
    gradient = jnp.array([0.5, 0.0, -0.4, 0.0])
    linear = matrix @ solution - gradient
    bounds = alx.Bounds(
        lower=jnp.array([0.0, -2.0, -2.0, -2.0]),
        upper=jnp.array([2.0, 2.0, 1.0, 2.0]),
    )

    def fn(y, args):
        hessian, linear_term = args
        return 0.5 * y @ hessian @ y - linear_term @ y, None

    return fn, (matrix, linear), bounds, solution


def solve(fn, solver, y0, bounds, args=None, max_steps=200):
    """Minimise `fn` with the bounds given in the options."""
    options = {} if bounds is None else {"bounds": bounds.expand(y0)}
    return optx.minimise(
        fn, solver, y0, args, options=options, has_aux=True, max_steps=max_steps
    )


def test_newton_declares_that_it_supports_bounds():
    assert alx.Newton.supports_bounds is True
    assert alx.Newton().supports_bounds is True


@pytest.mark.parametrize("hessian", HESSIAN_MODES)
@pytest.mark.parametrize("start", [jnp.zeros(4), jnp.array([3.0, -3.0, 3.0, 3.0])])
def test_convex_quadratic_reaches_the_bounded_solution(hessian, start):
    fn, args, bounds, expected = quadratic_problem()
    solver = alx.Newton(rtol=1e-12, atol=1e-12, hessian=hessian)
    solution = solve(fn, solver, start, bounds, args)
    np.testing.assert_allclose(solution.value, expected, atol=1e-8)


@pytest.mark.parametrize("hessian", HESSIAN_MODES)
def test_rosenbrock_in_a_box_that_cuts_off_the_minimiser(hessian):
    # The unconstrained minimiser is (1, 1), and the best point with y[0] <= 0.5
    # has y[1] = y[0] ** 2.
    bounds = alx.Bounds(upper=jnp.array([0.5, 2.0]))
    solver = alx.Newton(rtol=1e-10, atol=1e-10, hessian=hessian)
    solution = solve(rosenbrock, solver, jnp.array([-1.2, 1.0]), bounds)
    np.testing.assert_allclose(solution.value, [0.5, 0.25], atol=1e-6)


@pytest.mark.parametrize("hessian", HESSIAN_MODES)
def test_rosenbrock_in_a_box_with_two_active_bounds(hessian):
    bounds = alx.Bounds(lower=jnp.array([-2.0, -2.0]), upper=jnp.array([0.5, 0.2]))
    solver = alx.Newton(rtol=1e-10, atol=1e-10, hessian=hessian)
    y0 = jnp.array([-1.2, 1.0])
    solution = solve(rosenbrock, solver, y0, bounds)
    expanded = bounds.expand(y0)
    gradient = jax.grad(lambda y: rosenbrock(y, None)[0])(solution.value)
    stationarity = solution.value - jnp.clip(
        solution.value - gradient, expanded.lower, expanded.upper
    )
    assert float(jnp.max(jnp.abs(stationarity))) < 1e-6
    # The second bound is active, and the negative gradient points out of the box
    # there.
    np.testing.assert_allclose(solution.value[1], 0.2, atol=1e-12)
    assert float(gradient[1]) < 0.0


@pytest.mark.parametrize("hessian", HESSIAN_MODES)
def test_an_inactive_bound_gives_the_unbounded_solution(hessian):
    solver = alx.Newton(rtol=1e-10, atol=1e-10, hessian=hessian)
    y0 = jnp.array([-1.2, 1.0])
    bounds = alx.Bounds(lower=jnp.array(-5.0), upper=jnp.array(5.0))
    unbounded = solve(rosenbrock, solver, y0, None)
    bounded = solve(rosenbrock, solver, y0, bounds)
    np.testing.assert_allclose(bounded.value, unbounded.value, atol=1e-8)
    np.testing.assert_allclose(bounded.value, [1.0, 1.0], atol=1e-6)
    steps = int(bounded.stats["num_steps"])
    assert steps <= int(unbounded.stats["num_steps"]) + 5


def test_the_iterates_stay_inside_the_box_and_the_gradient_is_not_projected():
    fn, args, bounds, expected = quadratic_problem()
    solver = alx.Newton(rtol=1e-12, atol=1e-12)
    start = jnp.array([3.0, -3.0, 3.0, 3.0])
    options = {"bounds": bounds.expand(start)}
    state = solver.init(
        fn,
        start,
        args,
        options,
        jax.ShapeDtypeStruct((), start.dtype),
        None,
        frozenset(),
    )

    @jax.jit
    def step(y, state):
        return solver.step(fn, y, args, options, state, frozenset())

    y = start
    for _ in range(30):
        y, state, _ = step(y, state)
        assert bool(jnp.all(y >= options["bounds"].lower))
        assert bool(jnp.all(y <= options["bounds"].upper))
        assert bool(jnp.all(state.y_eval >= options["bounds"].lower))
        assert bool(jnp.all(state.y_eval <= options["bounds"].upper))
    np.testing.assert_allclose(y, expected, atol=1e-8)
    # The gradient of the accepted point is the one of the function, which is not
    # zero at the active bounds.
    np.testing.assert_allclose(
        state.f_info.grad, jax.grad(lambda point: fn(point, args)[0])(y), atol=1e-12
    )
    assert float(jnp.abs(state.f_info.grad[0])) > 0.1


def test_a_fixed_variable_does_not_stall_the_stopping_test():
    # The first variable is at its bound and its gradient is large at every
    # iterate, so the solve must stop on the change of the free variable.
    def fn(y, args):
        return 100.0 * y[0] + 0.5 * jnp.sum((y[1:] - 2.0) ** 2), None

    bounds = alx.Bounds(lower=jnp.zeros(3))
    solution = solve(fn, alx.Newton(), jnp.ones(3), bounds)
    np.testing.assert_allclose(solution.value, [0.0, 2.0, 2.0], atol=1e-7)
    assert int(solution.stats["num_steps"]) < 20


@pytest.mark.parametrize("seed", [5, 10, 15])
def test_a_fixed_variable_stays_exactly_at_its_bound(seed):
    # The variables have scales that differ by four orders of magnitude, so the
    # linear solve for the free variables has rounding errors. The third variable
    # is at its lower bound and the negative gradient points out of the box there.
    # Its step must be exactly zero and not rounding noise, or the variable would
    # move off the bound and the projected gradient would count it as a free
    # variable.
    scale = 10.0 ** jnp.array([2.0, -2.0, 1.0, -1.0])
    factor = jax.random.normal(jax.random.key(seed), (4, 4))
    hessian = (factor @ factor.T + jnp.eye(4)) * jnp.outer(scale, scale)
    linear = jnp.array([1.0, 0.5, 37.0, 0.2]) * scale

    def fn(y, args):
        return 0.5 * y @ hessian @ y + linear @ y, None

    bounds = alx.Bounds(
        lower=jnp.array([-10.0, -10.0, 0.0, -10.0]) / scale,
        upper=jnp.full(4, 10.0) / scale,
    )
    start = jnp.array([0.1, 0.1, 0.0, 0.1]) / scale
    solution = solve(fn, alx.Newton(rtol=1e-10, atol=1e-10), start, bounds)
    assert float(solution.value[2]) == 0.0


def test_a_variable_with_equal_bounds_is_fixed():
    def fn(y, args):
        return jnp.sum((y - 2.0) ** 2), None

    bounds = alx.Bounds(
        lower=jnp.array([1.0, -5.0, -5.0]), upper=jnp.array([1.0, 5.0, 5.0])
    )
    solution = solve(fn, alx.Newton(), jnp.zeros(3), bounds)
    np.testing.assert_allclose(solution.value, [1.0, 2.0, 2.0], atol=1e-8)


def test_the_solve_is_jittable():
    fn, args, bounds, expected = quadratic_problem()
    solver = alx.Newton(rtol=1e-12, atol=1e-12)

    @jax.jit
    def run(upper):
        y0 = jnp.zeros(4)
        options = {"bounds": alx.Bounds(bounds.lower, upper).expand(y0)}
        return optx.minimise(
            fn, solver, y0, args, options=options, has_aux=True, max_steps=100
        ).value

    np.testing.assert_allclose(run(bounds.upper), expected, atol=1e-8)


def test_batched_bounds():
    fn, args, bounds, _ = quadratic_problem()
    solver = alx.Newton(rtol=1e-12, atol=1e-12)
    y0 = jnp.zeros(4)
    # The third upper bound is 1.0 and active in the first solve, 0.2 and active in
    # the second, and 5.0 and inactive in the third.
    thirds = jnp.array([1.0, 0.2, 5.0])
    uppers = jnp.broadcast_to(bounds.upper, (3, 4)).at[:, 2].set(thirds)

    def run(upper):
        options = {"bounds": alx.Bounds(bounds.lower, upper).expand(y0)}
        return optx.minimise(
            fn, solver, y0, args, options=options, has_aux=True, max_steps=100
        ).value

    batched = jax.vmap(run)(uppers)
    assert batched.shape == (3, 4)
    for index in range(3):
        np.testing.assert_allclose(batched[index], run(uppers[index]), atol=1e-8)
    np.testing.assert_allclose(batched[1][2], 0.2, atol=1e-12)
    assert float(batched[2][2]) < 5.0


def test_float32_solve():
    fn, args, bounds, expected = quadratic_problem()
    start = jnp.zeros(4, jnp.float32)
    float32_bounds = alx.Bounds(
        lower=bounds.lower.astype(jnp.float32), upper=bounds.upper.astype(jnp.float32)
    )

    float32_args = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), args)
    solution = solve(
        fn,
        alx.Newton(rtol=1e-5, atol=1e-5),
        start,
        float32_bounds,
        float32_args,
    )
    assert solution.value.dtype == jnp.float32
    np.testing.assert_allclose(solution.value, expected, atol=1e-4)


def test_float32_rosenbrock_in_a_box():
    bounds = alx.Bounds(upper=jnp.array([0.5, 2.0], jnp.float32))

    def float32_rosenbrock(y, args):
        return rosenbrock(y, args)[0].astype(jnp.float32), None

    solution = solve(
        float32_rosenbrock,
        alx.Newton(rtol=1e-5, atol=1e-5),
        jnp.array([-1.2, 1.0], jnp.float32),
        bounds,
    )
    assert solution.value.dtype == jnp.float32
    np.testing.assert_allclose(solution.value, [0.5, 0.25], atol=1e-3)


def test_pytree_variables_with_pytree_bounds():
    def fn(y, args):
        value = jnp.sum((y["a"] - 3.0) ** 2) + (y["b"] + 1.0) ** 2
        return value, None

    y0 = {"a": jnp.zeros(2), "b": jnp.array(0.0)}
    bounds = alx.Bounds(
        lower={"a": jnp.array([-1.0, -1.0]), "b": jnp.array(-0.5)},
        upper={"a": jnp.array([2.0, 5.0]), "b": jnp.array(1.0)},
    )
    solution = solve(fn, alx.Newton(), y0, bounds)
    np.testing.assert_allclose(solution.value["a"], [2.0, 3.0], atol=1e-8)
    np.testing.assert_allclose(solution.value["b"], -0.5, atol=1e-8)


def test_a_starting_point_outside_the_box_is_projected():
    def fn(y, args):
        return jnp.sum((y - 1.0) ** 2), None

    bounds = alx.Bounds(lower=jnp.zeros(2), upper=jnp.full(2, 3.0))
    solution = solve(fn, alx.Newton(), jnp.array([-4.0, 9.0]), bounds)
    np.testing.assert_allclose(solution.value, [1.0, 1.0], atol=1e-8)


@pytest.mark.parametrize("mode", ["dense", "cholesky", "matrix_free", "sparse"])
def test_restricted_step_matches_the_reduced_problem(mode):
    if mode == "sparse" and importlib.util.find_spec("splineax") is None:
        pytest.skip("Install the `sparse` dependency group.")
    size = 6
    key = jax.random.key(0)
    factor = jax.random.normal(key, (size, size))
    matrix = factor @ factor.T + size * jnp.eye(size)
    # A banded matrix has few nonzero entries, so the sparse pattern is small.
    matrix = jnp.where(
        jnp.abs(jnp.arange(size)[:, None] - jnp.arange(size)) <= 2, matrix, 0.0
    )
    matrix = matrix + 10.0 * jnp.eye(size)
    gradient = jnp.arange(1.0, size + 1.0)
    free = jnp.array([True, False, True, True, False, True])

    def scalar(point):
        return 0.5 * point @ matrix @ point

    strategy = {
        "dense": alx.DenseHessian(),
        "cholesky": alx.DenseHessian(eigendecomposition_limit=1),
        "matrix_free": alx.MatrixFreeHessian(),
        "sparse": alx.SparseHessian(),
    }[mode]
    prepared = strategy.prepare(
        lambda y, args: (scalar(y), None), jnp.zeros(size), None
    )
    data = prepared.evaluate(scalar, jnp.zeros(size))
    step = prepared.step(scalar, data, gradient, jnp.asarray(1e3), free)
    indices = np.flatnonzero(np.asarray(free))
    reduced = -jnp.linalg.solve(matrix[np.ix_(indices, indices)], gradient[indices])
    # Conjugate gradients stop at a loose tolerance, so the step only approximates
    # the reduced solution.
    tolerance = 0.2 if mode == "matrix_free" else 1e-5
    np.testing.assert_allclose(step[indices], reduced, rtol=tolerance, atol=1e-8)
    np.testing.assert_array_equal(step[np.flatnonzero(~np.asarray(free))], 0.0)


def test_without_bounds_in_the_options_the_unbounded_method_runs():
    solution = solve(
        rosenbrock, alx.Newton(rtol=1e-10, atol=1e-10), jnp.array([-1.2, 1.0]), None
    )
    np.testing.assert_allclose(solution.value, [1.0, 1.0], atol=1e-7)
    assert int(solution.stats["num_steps"]) < 40
