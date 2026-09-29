import jax.numpy as jnp
import numpy as np

import aulax as alx
from aulax._lagrangian import (
    augmented_lagrangian,
    kkt_infeasibility,
    lagrangian,
    penalty_infeasibility,
)


EVALUATION = alx.Evaluation(
    objective=jnp.array(2.0),
    equality={"a": jnp.array([0.5, -1.0]), "b": jnp.array(0.25)},
    inequality=jnp.array([-2.0, 0.0, 0.5]),
)
MULTIPLIERS = alx.Constraints(
    equality={"a": jnp.array([1.0, 2.0]), "b": jnp.array(-1.0)},
    inequality=jnp.array([0.5, 1.0, 0.0]),
)
PENALTY = alx.Constraints(
    equality={"a": jnp.array([10.0, 10.0]), "b": jnp.array(4.0)},
    inequality=jnp.array([10.0, 2.0, 10.0]),
)


def flat(tree) -> np.ndarray:
    """Return the equality and inequality leaves as one flat array."""
    return np.concatenate(
        [np.ravel(leaf) for leaf in (tree["a"], tree["b"])]
        if isinstance(tree, dict)
        else [np.ravel(tree)]
    )


def test_augmented_lagrangian_matches_formula():
    h, lam, rho_h = (
        flat(EVALUATION.equality),
        flat(MULTIPLIERS.equality),
        flat(PENALTY.equality),
    )
    g, mu, rho_g = (
        flat(EVALUATION.inequality),
        flat(MULTIPLIERS.inequality),
        flat(PENALTY.inequality),
    )
    expected = (
        2.0
        + np.sum(lam * h + 0.5 * rho_h * h**2)
        + np.sum((np.maximum(0.0, mu + rho_g * g) ** 2 - mu**2) / (2 * rho_g))
    )
    assert np.isclose(augmented_lagrangian(EVALUATION, MULTIPLIERS, PENALTY), expected)


def test_augmented_lagrangian_equals_shifted_penalty_form_up_to_constant():
    h, lam, rho_h = (
        flat(EVALUATION.equality),
        flat(MULTIPLIERS.equality),
        flat(PENALTY.equality),
    )
    g, mu, rho_g = (
        flat(EVALUATION.inequality),
        flat(MULTIPLIERS.inequality),
        flat(PENALTY.inequality),
    )
    shifted = (
        2.0
        + np.sum(0.5 * rho_h * (h + lam / rho_h) ** 2)
        + np.sum(0.5 * rho_g * np.maximum(0.0, g + mu / rho_g) ** 2)
    )
    constant = np.sum(lam**2 / (2 * rho_h)) + np.sum(mu**2 / (2 * rho_g))
    assert np.isclose(
        augmented_lagrangian(EVALUATION, MULTIPLIERS, PENALTY), shifted - constant
    )


def test_augmented_lagrangian_without_constraints_is_objective():
    evaluation = alx.Evaluation(objective=jnp.array(3.0))
    empty = alx.Constraints(equality=None, inequality=None)
    assert augmented_lagrangian(evaluation, empty, empty) == 3.0
    assert lagrangian(evaluation, empty) == 3.0
    assert kkt_infeasibility(evaluation, empty) == 0.0


def test_lagrangian_matches_formula():
    expected = (
        2.0
        + np.dot(flat(MULTIPLIERS.equality), flat(EVALUATION.equality))
        + np.dot(flat(MULTIPLIERS.inequality), flat(EVALUATION.inequality))
    )
    assert np.isclose(lagrangian(EVALUATION, MULTIPLIERS), expected)


def test_penalty_infeasibility_is_elementwise():
    infeasibility = penalty_infeasibility(EVALUATION, MULTIPLIERS, PENALTY)
    np.testing.assert_allclose(infeasibility.equality["a"], [0.5, 1.0])
    np.testing.assert_allclose(infeasibility.equality["b"], 0.25)
    # The first constraint is inactive with a small multiplier, the second holds
    # with equality, and the third is violated.
    np.testing.assert_allclose(infeasibility.inequality, [0.05, 0.0, 0.5])


def test_kkt_infeasibility_takes_largest_violation():
    # The largest term is the complementarity of the first inequality, which is
    # inactive at -2 with a multiplier of 0.5.
    assert np.isclose(kkt_infeasibility(EVALUATION, MULTIPLIERS), 1.0)
