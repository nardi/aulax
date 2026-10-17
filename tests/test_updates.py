import jax.numpy as jnp
import numpy as np

import aulax as alx
from aulax import (
    AlgencanPenaltyUpdate,
    GeometricToleranceSchedule,
    SafeguardedMultiplierUpdate,
)


EVALUATION = alx.Evaluation(
    objective=jnp.array(-30.0),
    equality=jnp.array([1.0, -2.0]),
    inequality=jnp.array([-1.0, 3.0]),
)


def constraints(equality, inequality) -> alx.Constraints:
    return alx.Constraints(
        equality=jnp.asarray(equality, dtype=float),
        inequality=jnp.asarray(inequality, dtype=float),
    )


def test_multiplier_update_is_first_order():
    update = SafeguardedMultiplierUpdate()
    multipliers = constraints([0.5, 0.5], [2.0, 0.0])
    penalty = constraints([10.0, 10.0], [1.0, 1.0])
    estimates = update.update_multipliers(multipliers, penalty, EVALUATION)
    np.testing.assert_allclose(estimates.equality, [10.5, -19.5])
    np.testing.assert_allclose(estimates.inequality, [1.0, 3.0])


def test_multiplier_safeguard_clips_to_box():
    update = SafeguardedMultiplierUpdate(equality_bound=5.0, inequality_bound=2.0)
    multipliers = constraints([10.0, -10.0], [3.0, 1.0])
    safeguarded = update.safeguard_multipliers(multipliers, EVALUATION)
    np.testing.assert_allclose(safeguarded.equality, [5.0, -5.0])
    np.testing.assert_allclose(safeguarded.inequality, [2.0, 1.0])


def test_multiplier_bound_defaults_depend_on_dtype():
    update = SafeguardedMultiplierUpdate()
    multipliers = constraints([1e10, 0.0], [1e10, 0.0])
    float32_evaluation = alx.Evaluation(objective=jnp.array(0.0, dtype=jnp.float32))
    np.testing.assert_allclose(
        update.safeguard_multipliers(multipliers, float32_evaluation).equality,
        [1e6, 0.0],
    )
    np.testing.assert_allclose(
        update.safeguard_multipliers(multipliers, EVALUATION).equality, [1e10, 0.0]
    )


def test_initial_penalty_follows_algencan_formula():
    penalty = AlgencanPenaltyUpdate().initial_penalty(EVALUATION)
    # The squared violation is (1 + 4 + 9) / 2 = 7 and the objective is -30.
    np.testing.assert_allclose(penalty.equality, [300.0 / 7.0] * 2)
    np.testing.assert_allclose(penalty.inequality, [300.0 / 7.0] * 2)


def test_initial_penalty_can_be_fixed():
    penalty = AlgencanPenaltyUpdate(initial_value=3.0).initial_penalty(EVALUATION)
    np.testing.assert_allclose(penalty.inequality, [3.0, 3.0])


def test_penalty_grows_only_without_enough_decrease():
    update = AlgencanPenaltyUpdate()
    penalty = constraints([1.0, 1.0], [1.0, 1.0])
    previous = constraints([1.0, 1.0], [1.0, 1.0])
    decreased, _ = update.update_penalty(
        penalty, constraints([0.4, 0.1], [0.0, 0.5]), previous
    )
    np.testing.assert_allclose(decreased.equality, [1.0, 1.0])
    stagnated, stalled = update.update_penalty(
        penalty, constraints([0.4, 0.1], [0.0, 0.6]), previous
    )
    np.testing.assert_allclose(stagnated.equality, [10.0, 10.0])
    np.testing.assert_allclose(stagnated.inequality, [10.0, 10.0])
    assert not stalled


def test_per_constraint_penalty_grows_elementwise():
    update = AlgencanPenaltyUpdate(per_constraint=True)
    penalty = constraints([1.0, 1.0], [1.0, 1.0])
    previous = constraints([1.0, 1.0], [1.0, 1.0])
    new_penalty, _ = update.update_penalty(
        penalty, constraints([0.4, 0.9], [0.0, 0.6]), previous
    )
    np.testing.assert_allclose(new_penalty.equality, [1.0, 10.0])
    np.testing.assert_allclose(new_penalty.inequality, [1.0, 10.0])


def test_penalty_does_not_grow_below_the_infeasibility_floor():
    update = AlgencanPenaltyUpdate(infeasibility_floor=1e-6)
    penalty = constraints([1.0, 1.0], [1.0, 1.0])
    # The infeasibility did not halve, but it is already at rounding level.
    previous = constraints([1e-9, 1e-9], [1e-9, 1e-9])
    same, stalled = update.update_penalty(penalty, previous, previous)
    np.testing.assert_allclose(same.equality, [1.0, 1.0])
    assert not stalled


def test_penalty_stops_at_maximum_and_reports_stall():
    update = AlgencanPenaltyUpdate(maximum_value=5.0)
    previous = constraints([1.0, 1.0], [1.0, 1.0])
    infeasibility = constraints([1.0, 1.0], [1.0, 1.0])
    capped, stalled = update.update_penalty(
        constraints([1.0, 1.0], [1.0, 1.0]), infeasibility, previous
    )
    np.testing.assert_allclose(capped.equality, [5.0, 5.0])
    assert not stalled
    _, stalled = update.update_penalty(capped, infeasibility, previous)
    assert stalled


def test_geometric_schedule_shrinks_tolerance():
    schedule = GeometricToleranceSchedule(initial=1e-2, factor=0.1)
    assert schedule.initial_tolerance() == 1e-2
    np.testing.assert_allclose(schedule.update_tolerance(jnp.array(1e-2)), 1e-3)
