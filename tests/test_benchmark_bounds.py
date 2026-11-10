"""Tests for `benchmarks.bounds.split_bounds`."""

import jax.numpy as jnp
import numpy as np

import aulax as alx
from benchmarks.bounds import split_bounds


def bounded_problem(y, args, state):
    """Return a problem with bounds `y["x"][0] <= 1` and `y["x"][1] >= -2`.

    The general residual `y["x"][0] + y["z"] - 3 <= 0` is not a bound.
    """
    x = y["x"]
    z = y["z"]
    return alx.Evaluation(
        objective=jnp.sum(x**2) + z**2,
        inequality={
            "bounds": jnp.stack([x[0] - 1.0, -x[1] - 2.0]),
            "general": x[0] + z - 3.0,
        },
    )


def only_bounds_problem(y, args, state):
    """Return a problem whose inequality residuals are all bounds."""
    return alx.Evaluation(
        objective=jnp.sum(y**2),
        inequality=jnp.stack([y[0] - 1.0, -y[1] - 2.0]),
    )


Y0 = {"x": np.array([0.5, 0.0]), "z": np.array(0.0)}


def test_split_bounds_returns_box_with_structure_of_y0():
    bounds, _ = split_bounds(bounded_problem, Y0, None)
    assert isinstance(bounds, alx.Bounds)
    assert bounds.lower is not None and bounds.upper is not None
    np.testing.assert_array_equal(bounds.upper["x"], [1.0, np.inf])
    np.testing.assert_array_equal(bounds.lower["x"], [-np.inf, -2.0])
    assert bounds.upper["z"] == np.inf
    assert bounds.lower["z"] == -np.inf


def test_reduced_problem_keeps_only_the_general_residual():
    _, reduced = split_bounds(bounded_problem, Y0, None)
    evaluation = reduced(Y0, None, None)
    np.testing.assert_allclose(evaluation.inequality, [0.5 + 0.0 - 3.0])
    assert evaluation.inequality.shape == (1,)


def test_reduced_problem_without_general_residuals_has_no_inequalities():
    y0 = np.array([0.0, 0.0])
    bounds, reduced = split_bounds(only_bounds_problem, y0, None)
    evaluation = reduced(y0, None, None)
    assert evaluation.inequality is None
    np.testing.assert_array_equal(bounds.upper, [1.0, np.inf])
    np.testing.assert_array_equal(bounds.lower, [-np.inf, -2.0])


def test_problem_without_inequalities_is_returned_unchanged():
    def equality_only(y, args, state):
        return alx.Evaluation(objective=jnp.sum(y**2), equality=y[0] - 1.0)

    y0 = np.array([0.0, 0.0])
    bounds, reduced = split_bounds(equality_only, y0, None)
    assert reduced is equality_only
    np.testing.assert_array_equal(bounds.upper, [np.inf, np.inf])
    np.testing.assert_array_equal(bounds.lower, [-np.inf, -np.inf])
