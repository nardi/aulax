"""Checks for the optimistix behavior that the aulax solver loop relies on.

`AugmentedLagrangian` drives an optimistix minimiser one `step` at a time instead
of calling `optimistix.minimise`. That depends on details of `init` and `step`
that optimistix does not document as guarantees. An optimistix release that
changes one of them should fail here first.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from jaxtyping import Array, Float, Scalar


MINIMISERS = [
    optx.BFGS(rtol=1e-8, atol=1e-8),
    optx.LBFGS(rtol=1e-8, atol=1e-8),
    optx.NonlinearCG(rtol=1e-8, atol=1e-8),
    optx.GradientDescent(learning_rate=0.1, rtol=1e-8, atol=1e-8),
]


def quadratic_with_point_as_aux(
    y: Float[Array, " 2"], args: None
) -> tuple[Scalar, Float[Array, " 2"]]:
    """Return a convex quadratic, with the evaluated point as the aux output."""
    return jnp.sum((y - jnp.array([1.0, -2.0])) ** 2), y


def init_state(solver: optx.AbstractMinimiser, fn, y: Array, args):
    """Initialize an optimistix solver state the way `iterative_solve` does."""
    f_struct, aux_struct = jax.eval_shape(fn, y, args)
    return solver.init(fn, y, args, {}, f_struct, aux_struct, frozenset())


@pytest.mark.parametrize("solver", MINIMISERS, ids=lambda solver: type(solver).__name__)
def test_step_returns_aux_of_input_point(solver: optx.AbstractMinimiser):
    y = jnp.array([3.0, 4.0])
    state = init_state(solver, quadratic_with_point_as_aux, y, None)
    for _ in range(5):
        new_y, state, aux = solver.step(
            quadratic_with_point_as_aux, y, None, {}, state, frozenset()
        )
        # The aux output belongs to the point that went into `step`, not to the
        # accepted point that comes out of it.
        assert jnp.array_equal(aux, y)
        y = new_y


@pytest.mark.parametrize("solver", MINIMISERS, ids=lambda solver: type(solver).__name__)
def test_init_does_not_evaluate_function(solver: optx.AbstractMinimiser):
    evaluation_count = 0

    def counting_fn(y: Array, args: None) -> tuple[Scalar, Array]:
        nonlocal evaluation_count
        evaluation_count += 1
        return quadratic_with_point_as_aux(y, args)

    y = jnp.array([3.0, 4.0])
    f_struct, aux_struct = jax.eval_shape(quadratic_with_point_as_aux, y, None)
    solver.init(counting_fn, y, None, {}, f_struct, aux_struct, frozenset())
    assert evaluation_count == 0


@pytest.mark.parametrize("solver", MINIMISERS, ids=lambda solver: type(solver).__name__)
def test_step_accepts_changing_args(solver: optx.AbstractMinimiser):
    def shifted_quadratic(y: Array, args: Array) -> tuple[Scalar, None]:
        return jnp.sum((y - args) ** 2), None

    y = jnp.array([3.0, 4.0])
    state = init_state(solver, shifted_quadratic, y, jnp.zeros(2))
    structure = jax.tree.structure(state)
    for shift in range(3):
        args = jnp.full(2, float(shift))
        y, state, _ = solver.step(shifted_quadratic, y, args, {}, state, frozenset())
        assert jax.tree.structure(state) == structure


@pytest.mark.parametrize("solver", MINIMISERS, ids=lambda solver: type(solver).__name__)
def test_state_splits_into_arrays_and_static_parts(solver: optx.AbstractMinimiser):
    y = jnp.array([3.0, 4.0])
    state = init_state(solver, quadratic_with_point_as_aux, y, None)
    dynamic_state, static_state = eqx.partition(state, eqx.is_array)
    _, stepped_state, _ = solver.step(
        quadratic_with_point_as_aux, y, None, {}, state, frozenset()
    )
    stepped_dynamic_state, _ = eqx.partition(stepped_state, eqx.is_array)
    reinitialized_state = init_state(solver, quadratic_with_point_as_aux, y + 1.0, None)
    reinitialized_dynamic_state, _ = eqx.partition(reinitialized_state, eqx.is_array)
    # A loop carry can only hold arrays, so the solver loop carries the array
    # part and keeps the static part from the first `init`. That works as long as
    # `step` and a later `init` produce array parts with the same structure.
    assert jax.tree.structure(stepped_dynamic_state) == jax.tree.structure(
        dynamic_state
    )
    assert jax.tree.structure(reinitialized_dynamic_state) == jax.tree.structure(
        dynamic_state
    )
    combined_state = eqx.combine(reinitialized_dynamic_state, static_state)
    solver.step(quadratic_with_point_as_aux, y, None, {}, combined_state, frozenset())


@pytest.mark.parametrize("solver", MINIMISERS, ids=lambda solver: type(solver).__name__)
def test_traced_tolerances_compile_once(solver: optx.AbstractMinimiser):
    trace_count = 0

    @jax.jit
    def solve(tolerance: Scalar) -> Array:
        nonlocal trace_count
        trace_count += 1
        traced_solver = eqx.tree_at(
            lambda solver: (solver.rtol, solver.atol), solver, (tolerance, tolerance)
        )
        y = jnp.array([3.0, 4.0])
        state = init_state(traced_solver, quadratic_with_point_as_aux, y, None)
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)

        def body(_, carry):
            y, dynamic_state = carry
            state = eqx.combine(dynamic_state, static_state)
            y, state, _ = traced_solver.step(
                quadratic_with_point_as_aux, y, None, {}, state, frozenset()
            )
            dynamic_state, _ = eqx.partition(state, eqx.is_array)
            return y, dynamic_state

        y, _ = jax.lax.fori_loop(0, 100, body, (y, dynamic_state))
        return y

    loose = solve(jnp.array(1e-2))
    tight = solve(jnp.array(1e-10))
    assert trace_count == 1
    assert jnp.allclose(tight, jnp.array([1.0, -2.0]), atol=1e-6)
    assert jnp.all(jnp.isfinite(loose))
