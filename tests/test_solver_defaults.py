import jax.numpy as jnp
import optimistix as optx

import aulax as alx


def test_lbfgs_selects_a_larger_penalty_and_keeps_its_history():
    solver = alx.AugmentedLagrangian(inner=alx.LBFGS())
    penalty_update = solver.penalty_update_for_inner()
    assert isinstance(penalty_update, alx.AlgencanPenaltyUpdate)
    assert penalty_update.initial_value == 10.0
    assert penalty_update.growth_factor == 100.0
    state_access = solver.inner_state_access_for_inner()
    assert isinstance(state_access, alx.OptimistixStateAccess)
    assert state_access.hessian_reuse == "keep_unless_penalty_increased"
    schedule = solver.tolerance_schedule_for_inner()
    assert schedule.initial_tolerance() == 1e-8


def test_newton_and_other_minimisers_keep_the_generic_defaults():
    for inner in (alx.Newton(), optx.BFGS(rtol=1e-8, atol=1e-8)):
        solver = alx.AugmentedLagrangian(inner=inner)
        penalty_update = solver.penalty_update_for_inner()
        assert isinstance(penalty_update, alx.AlgencanPenaltyUpdate)
        assert penalty_update.initial_value is None
        assert penalty_update.growth_factor == 10.0
        state_access = solver.inner_state_access_for_inner()
        assert isinstance(state_access, alx.OptimistixStateAccess)
        assert state_access.hessian_reuse == "reset"
        assert solver.tolerance_schedule_for_inner().initial_tolerance() == 1e-2


def test_explicit_fields_replace_the_defaults_of_the_minimiser():
    penalty_update = alx.AlgencanPenaltyUpdate(initial_value=3.0)
    solver = alx.AugmentedLagrangian(inner=alx.LBFGS(), penalty_update=penalty_update)
    assert solver.penalty_update_for_inner() is penalty_update


def test_feasibility_tolerance_by_minimiser_and_dtype():
    newton = alx.AugmentedLagrangian(inner=alx.Newton())
    lbfgs = alx.AugmentedLagrangian(inner=alx.LBFGS())
    other = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8))
    assert newton.feasibility_tolerance_for(jnp.float64) == 1e-8
    assert lbfgs.feasibility_tolerance_for(jnp.float64) == 1e-8
    assert other.feasibility_tolerance_for(jnp.float64) == 1e-6
    assert newton.feasibility_tolerance_for(jnp.float32) == 1e-5
    assert lbfgs.feasibility_tolerance_for(jnp.float32) == 1e-5
    assert other.feasibility_tolerance_for(jnp.float32) == 1e-3


def test_inner_tolerance_floor_applies_only_in_float32_by_default():
    solver = alx.AugmentedLagrangian(inner=alx.LBFGS(rtol=1e-10, atol=1e-10))
    assert solver.final_inner_tolerance(jnp.float64) == 1e-10
    assert solver.final_inner_tolerance(jnp.float32) == 1e-6
    explicit = alx.AugmentedLagrangian(
        inner=alx.LBFGS(rtol=1e-10, atol=1e-10), inner_tolerance_floor=1e-7
    )
    assert explicit.final_inner_tolerance(jnp.float64) == 1e-7
