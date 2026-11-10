import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jaxtyping import Array, Float, Scalar
from problems import hs28

import aulax as alx
from aulax._lbfgs import _LBFGSState


def rosenbrock(y: Float[Array, " 2"], args: None) -> tuple[Scalar, None]:
    """Return the Rosenbrock function, which has its minimum at (1, 1)."""
    return (1.0 - y[0]) ** 2 + 100.0 * (y[1] - y[0] ** 2) ** 2, None


def quadratic(y: Float[Array, " n"], matrix_and_target) -> tuple[Scalar, None]:
    """Return a quadratic function of `y` with the minimiser `solve(matrix, target)`."""
    matrix, target = matrix_and_target
    return 0.5 * y @ matrix @ y - target @ y, None


def solve_with_bounds(
    function, solver, y0, bounds=None, args=None, max_steps=1000, **kwargs
):
    """Run `optx.minimise` with an expanded `bounds` in the options."""
    options = {} if bounds is None else {"bounds": bounds.expand(y0)}
    return optx.minimise(
        function,
        solver,
        y0,
        args,
        options=options,
        has_aux=True,
        max_steps=max_steps,
        throw=False,
        **kwargs,
    )


def test_unconstrained_quadratic():
    rng = np.random.default_rng(0)
    factor = rng.normal(size=(8, 8))
    matrix = jnp.asarray(factor @ factor.T + np.eye(8))
    target = jnp.asarray(rng.normal(size=8))
    solution = solve_with_bounds(
        quadratic,
        alx.LBFGS(rtol=1e-12, atol=1e-12),
        jnp.zeros(8),
        args=(matrix, target),
    )
    assert solution.result == optx.RESULTS.successful
    np.testing.assert_allclose(
        solution.value, jnp.linalg.solve(matrix, target), atol=1e-8
    )


def test_rosenbrock():
    solution = solve_with_bounds(
        rosenbrock, alx.LBFGS(rtol=1e-12, atol=1e-12), jnp.array([-1.2, 1.0])
    )
    assert solution.result == optx.RESULTS.successful
    np.testing.assert_allclose(solution.value, [1.0, 1.0], atol=1e-7)


@pytest.mark.parametrize("diagonal", [False, True])
def test_rosenbrock_with_each_initial_inverse_hessian(diagonal: bool):
    solver = alx.LBFGS(rtol=1e-12, atol=1e-12, diagonal=diagonal)
    solution = solve_with_bounds(rosenbrock, solver, jnp.array([-1.2, 1.0]))
    np.testing.assert_allclose(solution.value, [1.0, 1.0], atol=1e-7)


def test_state_has_gradient_and_aux_at_the_accepted_point():
    def function(y, args):
        return jnp.sum((y - 1.0) ** 4) + jnp.sum(y**2), jnp.sum(y)

    solution = optx.minimise(
        function,
        alx.LBFGS(rtol=1e-12, atol=1e-12),
        jnp.array([3.0, -2.0]),
        has_aux=True,
        max_steps=200,
        throw=False,
    )
    state = solution.state
    assert isinstance(state, _LBFGSState)
    np.testing.assert_allclose(
        state.f_info.grad, jax.grad(lambda y: function(y, None)[0])(solution.value)
    )
    np.testing.assert_allclose(state.f_info.f, function(solution.value, None)[0])
    np.testing.assert_allclose(state.aux, jnp.sum(solution.value))


def test_bounded_problem_ends_on_the_active_bounds():
    # The unconstrained minimiser is (2, -3, 0.5), and the box cuts off two of the
    # coordinates.
    target = jnp.array([2.0, -3.0, 0.5])

    def function(y, args):
        return jnp.sum((y - target) ** 2) + 0.1 * y[0] * y[1], None

    bounds = alx.Bounds(lower=jnp.array([-1.0, -1.0, -1.0]), upper=1.0)
    solution = solve_with_bounds(
        function, alx.LBFGS(rtol=1e-12, atol=1e-12), jnp.zeros(3), bounds
    )
    assert solution.result == optx.RESULTS.successful
    assert solution.value[0] == 1.0
    assert solution.value[1] == -1.0
    # The free coordinate solves its own equation, with the other two fixed.
    np.testing.assert_allclose(solution.value[2], 0.5, atol=1e-8)


def test_start_outside_the_box_and_iterates_stay_inside():
    target = jnp.array([4.0, -4.0])
    visited = []

    def function(y, args):
        jax.debug.callback(lambda point: visited.append(np.asarray(point)), y)
        return jnp.sum((y - target) ** 2), None

    bounds = alx.Bounds(lower=-1.0, upper=jnp.array([2.0, 2.0]))
    start = jnp.array([10.0, 10.0])
    solution = solve_with_bounds(
        function, alx.LBFGS(rtol=1e-12, atol=1e-12), start, bounds
    )
    np.testing.assert_array_equal(solution.value, [2.0, -1.0])
    points = np.stack(visited)
    assert np.all(points >= -1.0)
    assert np.all(points <= 2.0)


def test_pytree_variables_with_bounds():
    def function(y, args):
        value = (y["a"] - 2.0) ** 2 + jnp.sum((y["b"] + 1.0) ** 2) + y["a"] * 0.0
        return jnp.sum(value), None

    y0 = {"a": jnp.array(0.0), "b": jnp.zeros(2)}
    bounds = alx.Bounds(lower={"a": -5.0, "b": jnp.array([-0.5, -3.0])}, upper=None)
    solution = solve_with_bounds(
        function, alx.LBFGS(rtol=1e-12, atol=1e-12), y0, bounds
    )
    np.testing.assert_allclose(solution.value["a"], 2.0, atol=1e-8)
    assert solution.value["b"][0] == -0.5
    np.testing.assert_allclose(solution.value["b"][1], -1.0, atol=1e-8)


def test_pytree_variables_without_bounds():
    def function(y, args):
        return jnp.sum((y["a"] - 2.0) ** 2) + jnp.sum((y["b"] - y["a"][0]) ** 2), None

    y0 = {"a": jnp.zeros(2), "b": jnp.zeros((2, 2))}
    solution = solve_with_bounds(function, alx.LBFGS(rtol=1e-12, atol=1e-12), y0)
    np.testing.assert_allclose(solution.value["a"], 2.0, atol=1e-7)
    np.testing.assert_allclose(solution.value["b"], 2.0, atol=1e-7)


def test_batched_bounds_under_vmap_and_jit():
    def function(y, target):
        return jnp.sum((y - target) ** 2), None

    solver = alx.LBFGS(rtol=1e-12, atol=1e-12)
    y0 = jnp.zeros(2)
    target = jnp.array([3.0, -3.0])

    @jax.jit
    @jax.vmap
    def solve(upper):
        bounds = alx.Bounds(lower=-2.0, upper=upper).expand(y0)
        return optx.minimise(
            function,
            solver,
            y0,
            target,
            options={"bounds": bounds},
            has_aux=True,
            max_steps=100,
            throw=False,
        ).value

    upper = jnp.array([[1.0, 1.0], [2.0, 0.5], [10.0, 10.0]])
    values = solve(upper)
    expected = jnp.clip(target, -2.0, upper)
    np.testing.assert_array_equal(values, expected)


def test_variable_scale_is_the_box_width_where_both_bounds_are_finite():
    solver = alx.LBFGS()
    y0 = jnp.zeros(4)
    bounds = alx.Bounds(
        lower=jnp.array([0.0, -jnp.inf, 1.0, 2.0]),
        upper=jnp.array([10.0, 5.0, jnp.inf, 2.0]),
    ).expand(y0)
    state = solver.init(
        lambda y, args: (jnp.sum(y), None),
        y0,
        None,
        {"bounds": bounds},
        jax.ShapeDtypeStruct((), jnp.float64),
        None,
        frozenset(),
    )
    # The last variable has a box of width zero, which counts as unscaled.
    np.testing.assert_array_equal(state.variable_scale, [10.0, 1.0, 1.0, 1.0])
    unscaled = alx.LBFGS(variable_scaling="none").init(
        lambda y, args: (jnp.sum(y), None),
        y0,
        None,
        {"bounds": bounds},
        jax.ShapeDtypeStruct((), jnp.float64),
        None,
        frozenset(),
    )
    assert unscaled.variable_scale is None


@pytest.mark.parametrize("variable_scaling", ["bounds", "none"])
def test_boxes_of_very_different_widths(variable_scaling: str):
    target = jnp.array([4e5, 0.3, 2.0])

    def function(y, args):
        return jnp.sum(((y - target) / jnp.array([1e6, 1.0, 1.0])) ** 2), None

    bounds = alx.Bounds(lower=jnp.zeros(3), upper=jnp.array([1e6, 1.0, 1.0]))
    solver = alx.LBFGS(rtol=1e-12, atol=1e-12, variable_scaling=variable_scaling)
    solution = solve_with_bounds(function, solver, jnp.array([1e5, 0.9, 0.1]), bounds)
    np.testing.assert_allclose(solution.value[:2], [4e5, 0.3], rtol=1e-6)
    assert solution.value[2] == 1.0


def test_variable_scaling_must_be_known():
    with pytest.raises(ValueError, match="variable_scaling"):
        alx.LBFGS(variable_scaling="width")


def test_float32_with_and_without_bounds():
    y0 = jnp.array([-1.2, 1.0], dtype=jnp.float32)
    solver = alx.LBFGS(rtol=1e-6, atol=1e-6)
    solution = solve_with_bounds(rosenbrock, solver, y0)
    assert solution.value.dtype == jnp.float32
    assert solution.result == optx.RESULTS.successful
    np.testing.assert_allclose(solution.value, [1.0, 1.0], atol=1e-3)

    bounds = alx.Bounds(lower=-2.0, upper=jnp.array([0.5, 2.0]))
    solution = solve_with_bounds(rosenbrock, solver, y0, bounds)
    assert solution.value.dtype == jnp.float32
    assert solution.value[0] == jnp.float32(0.5)
    np.testing.assert_allclose(solution.value[1], 0.25, atol=1e-3)


def test_float32_state_dtypes():
    solver = alx.LBFGS()
    y0 = jnp.zeros(2, dtype=jnp.float32)
    state = solver.init(
        lambda y, args: (jnp.sum(y), None),
        y0,
        None,
        {},
        jax.ShapeDtypeStruct((), jnp.float32),
        None,
        frozenset(),
    )
    for leaf in jax.tree.leaves(state):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32


def augmented_lagrangian(**inner_options):
    """Return a solver with the settings that work for `alx.LBFGS` on small problems."""
    return alx.AugmentedLagrangian(
        inner=alx.LBFGS(rtol=1e-10, atol=1e-10, **inner_options),
        inner_max_steps=4096,
        tolerance_schedule=alx.GeometricToleranceSchedule(initial=1e-8),
        scaling=None,
        penalty_update=alx.AlgencanPenaltyUpdate(initial_value=1.0),
    )


@pytest.mark.parametrize(
    "hessian_reuse",
    ["reset", "keep", "keep_unless_penalty_increased", "keep_scaled"],
)
def test_augmented_lagrangian_with_equality_constraint(hessian_reuse: str):
    solver = eqx.tree_at(
        lambda current: current.inner_state_access,
        augmented_lagrangian(),
        alx.OptimistixStateAccess(hessian_reuse=hessian_reuse),
    )
    solution = alx.minimise(hs28, solver, jnp.zeros(3))
    np.testing.assert_allclose(solution.value, [0.5, -0.5, 0.5], atol=1e-5)
    assert solution.stats.infeasibility <= 1e-6


def make_state_with_history(count: int) -> _LBFGSState:
    """Return a state of `alx.LBFGS` after a few steps on a quadratic."""
    solver = alx.LBFGS(history_length=4)
    y0 = jnp.array([1.0, 2.0])
    state = solver.init(
        lambda y, args: (jnp.sum(y**2), None),
        y0,
        None,
        {},
        jax.ShapeDtypeStruct((), jnp.float64),
        None,
        frozenset(),
    )
    history = state.history
    history = history.__class__(
        steps=history.steps.at[:count].set(1.0),
        gradient_differences=history.gradient_differences.at[:count].set(2.0),
        count=jnp.array(count),
        index=jnp.array(count % 4),
        diagonal_numerator=None,
        diagonal_denominator=None,
    )
    return eqx.tree_at(lambda current: current.history, state, history)


@pytest.mark.parametrize(
    ("hessian_reuse", "penalty_ratio", "expected_count"),
    [
        ("reset", 1.0, 0),
        ("keep", 1.0, 3),
        ("keep", 10.0, 3),
        ("keep_unless_penalty_increased", 1.0, 3),
        ("keep_unless_penalty_increased", 10.0, 0),
        ("keep_scaled", 1.0, 3),
        ("keep_scaled", 10.0, 0),
    ],
)
def test_transfer_state_copies_the_history_as_the_reuse_option_says(
    hessian_reuse: str, penalty_ratio: float, expected_count: int
):
    old_state = make_state_with_history(3)
    new_state = make_state_with_history(0)
    access = alx.OptimistixStateAccess(hessian_reuse=hessian_reuse)
    transferred = access.transfer_state(
        old_state, new_state, jnp.asarray(penalty_ratio)
    )
    assert int(transferred.history.count) == expected_count
    if expected_count:
        np.testing.assert_array_equal(
            transferred.history.steps, old_state.history.steps
        )
    # The rest of the state is the one of the fresh solve.
    assert bool(transferred.first_step)


def test_minimiser_works_with_the_state_access_of_the_solver():
    access = alx.OptimistixStateAccess()
    y0 = jnp.array([-1.2, 1.0])
    solution = optx.minimise(
        rosenbrock,
        alx.LBFGS(rtol=1e-10, atol=1e-10),
        y0,
        has_aux=True,
        max_steps=200,
        throw=False,
    )
    gradient = access.accepted_gradient(solution.state)
    assert gradient is not None
    np.testing.assert_allclose(gradient, 0.0, atol=1e-5)
    assert access.accepted_evaluation(solution.state) is None
