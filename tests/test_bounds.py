from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jaxtyping import Array, Bool, Float, PyTree, Scalar

import aulax as alx
from aulax._misc import tree_max_norm


def test_expand_fills_missing_bounds_with_infinity():
    y = {"a": jnp.zeros(2), "b": jnp.zeros(())}
    expanded = alx.Bounds(lower={"a": 0.0, "b": -1.0}).expand(y)
    assert jnp.array_equal(expanded.lower["a"], jnp.zeros(2))
    assert expanded.lower["b"] == -1.0
    assert jnp.all(jnp.isposinf(expanded.upper["a"]))
    assert jnp.isposinf(expanded.upper["b"])


def test_expand_broadcasts_scalars_and_keeps_the_dtype_of_y():
    y = jnp.zeros((2, 3), jnp.float32)
    expanded = alx.Bounds(upper=2.0).expand(y)
    assert expanded.upper.shape == (2, 3)
    assert expanded.upper.dtype == jnp.float32


def test_clip_projects_onto_the_box():
    bounds = alx.Bounds(lower=jnp.array([0.0, -jnp.inf]), upper=jnp.array([1.0, 2.0]))
    clipped = bounds.clip(jnp.array([-3.0, 5.0]))
    assert jnp.array_equal(clipped, jnp.array([0.0, 2.0]))


class _ProjectedInfo(eqx.Module):
    """The value and gradient at the accepted point."""

    f: Scalar
    grad: PyTree[Float[Array, "..."]]


class _ProjectedState(eqx.Module):
    """The state of `ProjectedGradient` between steps."""

    f_info: _ProjectedInfo
    aux: PyTree
    step_size: Scalar
    terminate: Bool[Array, ""]


class ProjectedGradient(optx.AbstractMinimiser):
    """Gradient descent with projection onto the box and a backtracking line search.

    It is a minimal minimiser with native bounds, which tests the path of
    `alx.minimise` that hands the bounds to the inner minimiser. Every point that it
    evaluates is inside the box.
    """

    supports_bounds: ClassVar[bool] = True

    rtol: float = 1e-8
    atol: float = 1e-8
    norm: Callable[[PyTree], Scalar] = optx.max_norm

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        (value, aux), gradient = jax.value_and_grad(fn, has_aux=True)(y, args)
        return _ProjectedState(
            f_info=_ProjectedInfo(f=value, grad=gradient),
            aux=aux,
            step_size=jnp.ones((), value.dtype),
            terminate=jnp.array(False),
        )

    def step(self, fn, y, args, options, state, tags):
        # Without bounds, the box is all of the space.
        bounds = options.get("bounds", alx.Bounds()).expand(y)
        maximum_step = options.get("maximum_step", 1e6)
        value = state.f_info.f
        gradient = state.f_info.grad

        def trial(step_size):
            candidate = jax.tree.map(
                lambda leaf, slope, lower, upper: jnp.clip(
                    leaf - step_size * slope, lower, upper
                ),
                y,
                gradient,
                bounds.lower,
                bounds.upper,
            )
            (candidate_value, candidate_aux), candidate_gradient = jax.value_and_grad(
                fn, has_aux=True
            )(candidate, args)
            predicted = sum(
                jnp.sum(slope * (new - old))
                for slope, new, old in zip(
                    jax.tree.leaves(gradient),
                    jax.tree.leaves(candidate),
                    jax.tree.leaves(y),
                    strict=True,
                )
            )
            # Differences below the rounding error of the value are noise.
            noise = 8 * jnp.finfo(value.dtype).eps * jnp.abs(value)
            accepted = candidate_value <= value + 1e-4 * predicted + noise
            return (
                candidate,
                candidate_value,
                candidate_gradient,
                candidate_aux,
                accepted,
            )

        def shrink(carry):
            step_size, count, *_ = carry
            return (step_size / 2, count + 1, *trial(step_size / 2))

        def keep_shrinking(carry):
            _, count, *_, accepted = carry
            return jnp.invert(accepted) & (count < 40)

        first_step_size = jnp.minimum(2 * state.step_size, maximum_step)
        carry = (first_step_size, jnp.array(0), *trial(first_step_size))
        step_size, _, candidate, candidate_value, candidate_gradient, aux, accepted = (
            jax.lax.while_loop(keep_shrinking, shrink, carry)
        )
        new_y = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), candidate, y
        )
        new_value = jnp.where(accepted, candidate_value, value)
        new_gradient = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), candidate_gradient, gradient
        )
        new_aux = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), aux, state.aux
        )
        free_gradient = bounds.projected_gradient(new_y, new_gradient)
        converged = tree_max_norm(free_gradient) <= self.atol
        new_state = _ProjectedState(
            f_info=_ProjectedInfo(f=new_value, grad=new_gradient),
            aux=new_aux,
            step_size=step_size,
            terminate=converged | jnp.invert(accepted),
        )
        return new_y, new_state, state.aux

    def terminate(self, fn, y, args, options, state, tags):
        return state.terminate, optx.RESULTS.successful

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        return y, aux, {}


PROJECTED = alx.AugmentedLagrangian(
    inner=ProjectedGradient(rtol=1e-9, atol=1e-9),
    inner_stopping=alx.GradientStopping(),
    stationarity_tolerance=1e-5,
    inner_max_steps=2000,
)
LBFGS = alx.AugmentedLagrangian(inner=optx.LBFGS(rtol=1e-10, atol=1e-10))
NEWTON = alx.AugmentedLagrangian(inner=alx.Newton())
UNBOUNDED_NEWTON = alx.AugmentedLagrangian(inner=alx.Newton(rtol=1e-12, atol=1e-12))

# The solvers for float32 have looser tolerances, and `optimistix.LBFGS` does not
# run in float32 when 64-bit mode is on.
FLOAT32_SOLVERS = [
    pytest.param(
        alx.AugmentedLagrangian(
            inner=ProjectedGradient(rtol=1e-6, atol=1e-6),
            inner_stopping=alx.GradientStopping(),
            stationarity_tolerance=1e-4,
            inner_max_steps=2000,
        ),
        id="native",
    ),
    pytest.param(
        alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-6, atol=1e-6)), id="bfgs"
    ),
    pytest.param(NEWTON, id="newton"),
]

# The first solver passes the bounds to the minimiser, and the other two receive
# them as inequality constraints.
SOLVERS = [
    pytest.param(PROJECTED, id="native"),
    pytest.param(LBFGS, id="lbfgs"),
    pytest.param(NEWTON, id="newton"),
]


def minimise(*args, **kwargs):
    """Call `alx.minimise` with a step limit that the first-order minimiser needs."""
    kwargs.setdefault("max_steps", 50_000)
    return alx.minimise(*args, **kwargs)


def quadratic(y, target, state):
    return alx.Evaluation(objective=jnp.sum((y - target) ** 2))


QUADRATIC_TARGET = jnp.array([-1.0, 0.5, 3.0, 2.0])
QUADRATIC_BOUNDS = alx.Bounds(
    lower=jnp.array([0.0, -1.0, -jnp.inf, 0.0]),
    upper=jnp.array([jnp.inf, 1.0, 2.0, 2.5]),
)


@pytest.mark.parametrize("solver", SOLVERS)
def test_active_and_inactive_bounds(solver):
    solution = minimise(
        quadratic,
        solver,
        jnp.zeros(4),
        QUADRATIC_TARGET,
        bounds=QUADRATIC_BOUNDS,
    )
    np.testing.assert_allclose(solution.value, [0.0, 0.5, 2.0, 2.0], atol=1e-6)
    assert solution.bound_multipliers is not None
    # The gradient is 2 (y - target), which the first lower bound and the third
    # upper bound balance.
    np.testing.assert_allclose(
        solution.bound_multipliers.lower, [2.0, 0.0, 0.0, 0.0], atol=1e-5
    )
    np.testing.assert_allclose(
        solution.bound_multipliers.upper, [0.0, 0.0, 2.0, 0.0], atol=1e-5
    )
    assert solution.stats.stationarity < 1e-5
    assert solution.multipliers.equality is None
    assert solution.multipliers.inequality is None


@pytest.mark.parametrize("solver", SOLVERS)
def test_no_bounds_gives_no_bound_multipliers(solver):
    target = QUADRATIC_TARGET
    solution = minimise(quadratic, solver, jnp.zeros(4), target)
    assert solution.bound_multipliers is None
    empty = minimise(quadratic, solver, jnp.zeros(4), target, bounds=alx.Bounds())
    assert empty.bound_multipliers is None
    np.testing.assert_allclose(empty.value, target, atol=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_scalar_and_one_sided_bounds(solver):
    solution = minimise(
        quadratic,
        solver,
        jnp.ones(4),
        QUADRATIC_TARGET,
        bounds=alx.Bounds(lower=0.0),
    )
    np.testing.assert_allclose(solution.value, [0.0, 0.5, 3.0, 2.0], atol=1e-6)
    assert solution.bound_multipliers is not None
    # A field without bounds still has an entry for every variable.
    np.testing.assert_allclose(solution.bound_multipliers.upper, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        solution.bound_multipliers.lower, [2.0, 0.0, 0.0, 0.0], atol=1e-5
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_start_point_outside_the_box_is_moved_inside(solver):
    solution = minimise(
        quadratic,
        solver,
        jnp.array([50.0, -50.0, 50.0, -50.0]),
        QUADRATIC_TARGET,
        bounds=QUADRATIC_BOUNDS,
    )
    np.testing.assert_allclose(solution.value, [0.0, 0.5, 2.0, 2.0], atol=1e-6)


def test_lower_above_upper_is_rejected():
    bounds = alx.Bounds(lower=jnp.array([0.0, 1.0]), upper=jnp.array([1.0, 0.5]))
    with pytest.raises(ValueError, match="lower bound must be at most"):
        minimise(quadratic, LBFGS, jnp.zeros(2), jnp.ones(2), bounds=bounds)
    with pytest.raises(ValueError, match="lower bound must be at most"):
        minimise(quadratic, PROJECTED, jnp.zeros(2), jnp.ones(2), bounds=bounds)


def test_lower_above_upper_is_rejected_under_jit():
    @jax.jit
    def solve(lower):
        bounds = alx.Bounds(lower=lower, upper=jnp.zeros(2))
        return minimise(quadratic, LBFGS, jnp.zeros(2), jnp.ones(2), bounds=bounds)

    with pytest.raises(Exception, match="lower bound must be at most"):
        solve(jnp.array([0.0, 1.0]))


def test_options_must_not_contain_bounds():
    with pytest.raises(ValueError, match="`bounds`"):
        minimise(
            quadratic,
            PROJECTED,
            jnp.zeros(2),
            jnp.ones(2),
            bounds=alx.Bounds(lower=0.0),
            options={"bounds": None},
        )


def test_options_are_merged_with_the_bounds_of_a_native_minimiser():
    bounds = alx.Bounds(lower=0.0)
    solved = minimise(quadratic, PROJECTED, jnp.ones(2), jnp.ones(2), bounds=bounds)
    assert solved.result == alx.RESULTS.successful
    # A maximum step of zero stops the minimiser from moving, which shows that it
    # receives `options` and the bounds together.
    stuck = minimise(
        quadratic,
        PROJECTED,
        jnp.zeros(2),
        jnp.ones(2),
        bounds=bounds,
        options={"maximum_step": 0.0},
        throw=False,
    )
    np.testing.assert_allclose(stuck.value, 0.0)


def test_native_minimiser_only_evaluates_inside_the_box():
    def undefined_outside(y, args, state):
        inside = jnp.all((y >= 0.0) & (y <= 1.0))
        objective = jnp.sum((y - 2.0) ** 2)
        return alx.Evaluation(objective=jnp.where(inside, objective, jnp.nan))

    solution = minimise(
        undefined_outside,
        PROJECTED,
        jnp.full(3, 0.5),
        bounds=alx.Bounds(lower=0.0, upper=1.0),
    )
    np.testing.assert_allclose(solution.value, 1.0, atol=1e-8)
    assert solution.bound_multipliers is not None
    np.testing.assert_allclose(solution.bound_multipliers.upper, 2.0, atol=1e-5)


def constrained(y, parameter, state):
    """Return a problem with an equality and an inequality constraint."""
    return alx.Evaluation(
        objective=(y[0] - 2.0) ** 2 + (y[1] - 1.0) ** 2 + 0.5 * (y[2] + 1.0) ** 2,
        equality=y[0] + y[1] + y[2] - 1.5,
        inequality=jnp.array([y[0] - y[1] - 0.2, parameter - y[1]]),
    )


CONSTRAINED_BOUNDS = alx.Bounds(
    lower=jnp.array([0.0, 0.0, 0.0]), upper=jnp.array([1.0, 5.0, 5.0])
)


def constrained_with_rows(y, parameter, state):
    """Return `constrained` with the bounds written as inequality constraints."""
    evaluation = constrained(y, parameter, state)
    return alx.Evaluation(
        objective=evaluation.objective,
        equality=evaluation.equality,
        inequality=(
            evaluation.inequality,
            CONSTRAINED_BOUNDS.lower - y,
            y - CONSTRAINED_BOUNDS.upper,
        ),
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_bounds_with_equality_and_inequality_constraints(solver):
    reference = minimise(constrained_with_rows, UNBOUNDED_NEWTON, jnp.zeros(3), 0.1)
    solution = minimise(
        constrained,
        solver,
        jnp.zeros(3),
        0.1,
        bounds=CONSTRAINED_BOUNDS,
    )
    reference_lower = reference.multipliers.inequality[1]
    reference_upper = reference.multipliers.inequality[2]
    # The test is only meaningful if a bound and a constraint are active.
    assert jnp.any(reference_lower > 1e-3) or jnp.any(reference_upper > 1e-3)
    assert jnp.any(reference.multipliers.inequality[0] > 1e-3)
    np.testing.assert_allclose(solution.value, reference.value, atol=1e-5)
    np.testing.assert_allclose(
        solution.multipliers.equality, reference.multipliers.equality, atol=1e-4
    )
    np.testing.assert_allclose(
        solution.multipliers.inequality,
        reference.multipliers.inequality[0],
        atol=1e-4,
    )
    assert solution.bound_multipliers is not None
    np.testing.assert_allclose(
        solution.bound_multipliers.lower, reference_lower, atol=1e-4
    )
    np.testing.assert_allclose(
        solution.bound_multipliers.upper, reference_upper, atol=1e-4
    )
    # The multipliers of the user's constraints have the structure of the
    # constraints, with no entries for the bounds.
    assert jax.tree.structure(solution.multipliers) == jax.tree.structure(
        minimise(constrained, UNBOUNDED_NEWTON, jnp.zeros(3), 0.1).multipliers
    )
    assert solution.stats.infeasibility < 1e-5
    assert solution.stats.stationarity < 1e-4


@pytest.mark.parametrize("solver", SOLVERS)
def test_warm_start_with_multipliers(solver):
    first = minimise(constrained, solver, jnp.zeros(3), 0.1, bounds=CONSTRAINED_BOUNDS)
    second = minimise(
        constrained,
        solver,
        first.value,
        0.1,
        multipliers=first.multipliers,
        bounds=CONSTRAINED_BOUNDS,
    )
    np.testing.assert_allclose(second.value, first.value, atol=1e-5)


@pytest.mark.parametrize("solver", SOLVERS)
def test_pytree_variables_and_bounds(solver):
    def problem(y, target, state):
        return alx.Evaluation(
            objective=jnp.sum((y["pair"] - target["pair"]) ** 2)
            + (y["single"] - target["single"]) ** 2,
            equality=jnp.sum(y["pair"]) + y["single"] - 2.2,
        )

    target = {"pair": jnp.array([3.0, -1.0]), "single": jnp.array(0.5)}
    bounds = alx.Bounds(
        lower={"pair": jnp.array([-jnp.inf, 0.0]), "single": 0.0},
        upper={"pair": 2.0, "single": jnp.inf},
    )
    y0 = {"pair": jnp.zeros(2), "single": jnp.array(0.0)}
    solution = minimise(problem, solver, y0, target, bounds=bounds)
    reference = minimise(
        lambda y, args, state: alx.Evaluation(
            objective=problem(y, args, state).objective,
            equality=problem(y, args, state).equality,
            inequality={
                "pair_upper": y["pair"] - 2.0,
                "pair_lower": -y["pair"][1],
                "single": -y["single"],
            },
        ),
        UNBOUNDED_NEWTON,
        y0,
        target,
    )
    np.testing.assert_allclose(
        solution.value["pair"], reference.value["pair"], atol=1e-5
    )
    np.testing.assert_allclose(
        solution.value["single"], reference.value["single"], atol=1e-5
    )
    assert solution.bound_multipliers is not None
    np.testing.assert_allclose(
        solution.bound_multipliers.upper["pair"][0],
        reference.multipliers.inequality["pair_upper"][0],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        solution.bound_multipliers.lower["pair"][1],
        reference.multipliers.inequality["pair_lower"],
        atol=1e-4,
    )
    assert jax.tree.structure(solution.bound_multipliers.lower) == jax.tree.structure(
        y0
    )


@pytest.mark.parametrize("solver", FLOAT32_SOLVERS)
def test_float32_solve(solver):
    bounds = alx.Bounds(
        lower=jnp.array([0.0, -1.0, -jnp.inf, 0.0], jnp.float32),
        upper=jnp.array([jnp.inf, 1.0, 2.0, 2.5], jnp.float32),
    )
    solution = minimise(
        quadratic,
        solver,
        jnp.zeros(4, jnp.float32),
        QUADRATIC_TARGET.astype(jnp.float32),
        bounds=bounds,
    )
    assert solution.value.dtype == jnp.float32
    assert solution.bound_multipliers is not None
    assert solution.bound_multipliers.lower.dtype == jnp.float32
    np.testing.assert_allclose(solution.value, [0.0, 0.5, 2.0, 2.0], atol=1e-3)
    np.testing.assert_allclose(
        solution.bound_multipliers.lower, [2.0, 0.0, 0.0, 0.0], atol=5e-2
    )
    np.testing.assert_allclose(
        solution.bound_multipliers.upper, [0.0, 0.0, 2.0, 0.0], atol=5e-2
    )


@pytest.mark.parametrize("solver", FLOAT32_SOLVERS)
def test_float32_solve_with_constraints_and_python_bounds(solver):
    solution = minimise(
        constrained,
        solver,
        jnp.zeros(3, jnp.float32),
        jnp.asarray(0.1, jnp.float32),
        bounds=alx.Bounds(lower=0.0, upper=1.0),
    )
    assert solution.value.dtype == jnp.float32
    assert solution.stats.infeasibility < 1e-2
    # Bounds that are inequality constraints only hold to the feasibility tolerance.
    assert jnp.all(solution.value >= -1e-3)
    assert jnp.all(solution.value <= 1.0 + 1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
def test_jit_and_vmap_with_batched_start_and_bounds(solver):
    def solve(y0, lower, upper):
        solution = minimise(
            quadratic,
            solver,
            y0,
            QUADRATIC_TARGET,
            bounds=alx.Bounds(lower=lower, upper=upper),
        )
        return solution.value, solution.bound_multipliers

    y0 = jnp.array([[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, -5.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    lower = jnp.array([[0.0] * 4, [-2.0] * 4, [0.5] * 4])
    upper = jnp.array([[1.5] * 4, [4.0] * 4, [2.0] * 4])
    values, multipliers = jax.jit(jax.vmap(solve))(y0, lower, upper)
    for index in range(3):
        value, single = solve(y0[index], lower[index], upper[index])
        np.testing.assert_allclose(values[index], value, atol=1e-6)
        np.testing.assert_allclose(multipliers.lower[index], single.lower, atol=1e-5)
        np.testing.assert_allclose(
            values[index],
            jnp.clip(QUADRATIC_TARGET, lower[index], upper[index]),
            atol=1e-5,
        )
    # The third problem has the box [0.5, 2] and the target [-1, 0.5, 3, 2].
    np.testing.assert_allclose(multipliers.lower[2], [3.0, 0.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(multipliers.upper[2], [0.0, 0.0, 2.0, 0.0], atol=1e-4)


def parametrised(y, parameters, state):
    """Return a problem whose target and equality constraint depend on `parameters`."""
    return alx.Evaluation(
        objective=jnp.sum((y - parameters["target"]) ** 2) + 0.1 * y[0] * y[1] ** 2,
        equality=jnp.sum(y) - parameters["total"],
    )


def solve_parametrised(solver, parameters, bounds):
    solution = minimise(
        parametrised,
        solver,
        jnp.full(3, 0.3),
        parameters,
        bounds=bounds,
        max_steps=200_000,
    )
    return solution.value, solution.bound_multipliers, solution.multipliers


def finite_differences(function, point, step=1e-2):
    """Return the central differences of `function` along each entry of `point`."""
    columns = []
    for index in range(point.size):
        offset = jnp.zeros_like(point).at[index].set(step)
        columns.append(
            (function(point + offset) - function(point - offset)) / (2 * step)
        )
    return jnp.stack(columns, axis=-1)


@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(
            alx.AugmentedLagrangian(
                inner=ProjectedGradient(rtol=1e-6, atol=1e-6),
                inner_stopping=alx.GradientStopping(),
                stationarity_tolerance=1e-5,
                inner_max_steps=20_000,
                feasibility_tolerance=1e-7,
            ),
            id="native",
        ),
        pytest.param(
            alx.AugmentedLagrangian(inner=alx.Newton(rtol=1e-13, atol=1e-13)),
            id="newton",
        ),
    ],
)
@pytest.mark.parametrize(
    "upper_bound, active", [(0.9, True), (3.0, False)], ids=["active", "inactive"]
)
def test_derivatives_with_respect_to_problem_and_bounds_match_finite_differences(
    solver, upper_bound, active
):
    target = jnp.array([1.5, 0.5, 1.0])
    # The target of the first variable is above the upper bound when it is 0.9.
    point = jnp.concatenate([target, jnp.array([2.0, upper_bound])])

    def solve(point):
        parameters = {"target": point[:3], "total": point[3]}
        bounds = alx.Bounds(lower=0.0, upper=jnp.array([point[4], 5.0, 5.0]))
        value, reduced_costs, _ = solve_parametrised(solver, parameters, bounds)
        return jnp.concatenate([value, reduced_costs.upper])

    value_and_costs = solve(point)
    assert bool(value_and_costs[3] > 1e-3) == active
    jacobian = jax.jacfwd(solve)(point)
    reverse = jax.jacrev(solve)(point)
    expected = finite_differences(solve, point)
    np.testing.assert_allclose(jacobian, expected, atol=1e-3)
    np.testing.assert_allclose(reverse, expected, atol=1e-3)
    if active:
        # A variable at its upper bound moves one to one with the bound.
        np.testing.assert_allclose(jacobian[0, 4], 1.0, atol=1e-6)


@pytest.mark.parametrize("solver", [LBFGS, NEWTON], ids=["lbfgs", "newton"])
def test_fallback_derivatives_match_finite_differences(solver):
    target = jnp.array([1.5, 0.5, 1.0])
    point = jnp.concatenate([target, jnp.array([2.0, 0.9])])

    def solve(point):
        parameters = {"target": point[:3], "total": point[3]}
        bounds = alx.Bounds(lower=0.0, upper=jnp.array([point[4], 5.0, 5.0]))
        value, reduced_costs, _ = solve_parametrised(solver, parameters, bounds)
        return jnp.concatenate([value, reduced_costs.upper])

    expected = finite_differences(solve, point)
    np.testing.assert_allclose(jax.jacfwd(solve)(point), expected, atol=1e-3)


def test_projected_gradient_and_reduced_costs():
    bounds = alx.Bounds(
        lower=jnp.array([0.0, 0.0, 0.0, 0.0, -1.0]),
        upper=jnp.array([1.0, 1.0, 1.0, 1.0, -1.0]),
    )
    y = jnp.array([0.0, 0.0, 1.0, 0.5, -1.0])
    gradient = jnp.array([2.0, -3.0, 4.0, 1.5, 7.0])
    np.testing.assert_allclose(
        np.asarray(bounds.projected_gradient(y, gradient)), [0.0, -3.0, 4.0, 1.5, 0.0]
    )
    costs = bounds.reduced_costs(y, gradient)
    np.testing.assert_allclose(np.asarray(costs.lower), [2.0, 0.0, 0.0, 0.0, 7.0])
    np.testing.assert_allclose(np.asarray(costs.upper), [0.0, 0.0, 0.0, 0.0, 0.0])
    # The negated gradient makes the upper bound the active one.
    costs = bounds.reduced_costs(y, -gradient)
    np.testing.assert_allclose(np.asarray(costs.upper), [0.0, 0.0, 4.0, 0.0, 7.0])


def test_stationarity_is_measured_with_the_projected_gradient():
    solution = minimise(
        quadratic,
        PROJECTED,
        jnp.zeros(4),
        QUADRATIC_TARGET,
        bounds=QUADRATIC_BOUNDS,
    )
    gradient = jax.grad(lambda y: quadratic(y, QUADRATIC_TARGET, None).objective)(
        solution.value
    )
    # The plain gradient is large at the active bounds, and the statistic ignores
    # the part that the bounds balance.
    assert jnp.max(jnp.abs(gradient)) > 1.0
    assert solution.stats.stationarity < 1e-6


def test_gradient_stopping_uses_the_projected_gradient():
    solver = alx.AugmentedLagrangian(
        inner=ProjectedGradient(rtol=1e-3, atol=1e-3),
        inner_stopping=alx.GradientStopping(relative=False),
        stationarity_tolerance=1e-6,
        inner_max_steps=2000,
    )
    solution = minimise(
        quadratic,
        solver,
        jnp.zeros(4),
        QUADRATIC_TARGET,
        bounds=QUADRATIC_BOUNDS,
    )
    # The inner solve ends on the projected gradient. A test on the plain gradient
    # would never be satisfied at the active bounds.
    assert solution.result == alx.RESULTS.successful
    assert solution.stats.num_steps < 2000
    np.testing.assert_allclose(solution.value, [0.0, 0.5, 2.0, 2.0], atol=1e-5)


def test_equal_lower_and_upper_bound_fixes_the_variable():
    bounds = alx.Bounds(lower=jnp.array([0.0, 1.0]), upper=jnp.array([5.0, 1.0]))
    for solver in (PROJECTED, LBFGS):
        solution = minimise(
            quadratic, solver, jnp.zeros(2), jnp.array([2.0, 3.0]), bounds=bounds
        )
        np.testing.assert_allclose(solution.value, [2.0, 1.0], atol=1e-6)
        assert solution.bound_multipliers is not None
        np.testing.assert_allclose(
            solution.bound_multipliers.lower + solution.bound_multipliers.upper,
            [0.0, 4.0],
            atol=1e-2,
        )


def test_bounds_are_expanded_against_y0_and_clipped():
    bounds = alx.Bounds(lower=0.0, upper=jnp.array([1.0, 2.0]))
    expanded = bounds.expand(jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(expanded.lower), [0.0, 0.0])
    np.testing.assert_allclose(bounds.clip(jnp.array([-1.0, 5.0])), [0.0, 2.0])
    assert bool(bounds.is_ordered())
    assert not bool(alx.Bounds(lower=2.0, upper=jnp.array([1.0, 3.0])).is_ordered())
