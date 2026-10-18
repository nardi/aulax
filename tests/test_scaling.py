import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

import aulax as alx


def badly_scaled_projection(y, args, state):
    """Project `args` onto the simplex, with the constraints in other units.

    The sum constraint is multiplied by 1e4, which leaves
    the solution unchanged and its multiplier divided by 1e4.
    """
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2),
        equality=1e4 * (jnp.sum(y) - 1.0),
        inequality=-y,
    )


def solver(scaling):
    return alx.AugmentedLagrangian(
        inner=alx.Newton(rtol=1e-10, atol=1e-10),
        stationarity_tolerance=1e-6,
        scaling=scaling,
    )


POINT = jnp.array([0.6, 0.5, -0.4])


def test_gradient_scaling_factors():
    scaling = alx.GradientScaling(limit=100.0)

    def problem(y, args, state):
        return alx.Evaluation(
            objective=1e3 * jnp.sum(y),
            equality=jnp.array([1e4 * y[0], 5.0 * y[1]]),
            inequality=jnp.array([y[0] * y[1]]),
        )

    y0 = jnp.array([1.0, 2.0])
    evaluation = problem(y0, None, None)
    gradient = jax.grad(lambda y: problem(y, None, None).objective)(y0)
    factors = scaling.scale_factors(problem, y0, None, None, evaluation, gradient)
    np.testing.assert_allclose(factors.objective, 100.0 / 1e3)
    np.testing.assert_allclose(factors.constraints.equality, [100.0 / 1e4, 1.0])
    # The gradient of y0 * y1 at (1, 2) is (2, 1), which needs no scaling.
    np.testing.assert_allclose(factors.constraints.inequality, [1.0])


def test_scaling_keeps_solution_and_original_multipliers():
    plain = alx.minimise(
        badly_scaled_projection,
        solver(None),
        jnp.full(3, 1 / 3),
        POINT,
        throw=False,
    )
    scaled = alx.minimise(
        badly_scaled_projection,
        solver(alx.GradientScaling()),
        jnp.full(3, 1 / 3),
        POINT,
    )
    np.testing.assert_allclose(scaled.value, [0.55, 0.45, 0.0], atol=1e-6)
    # The multiplier of the sum constraint is 2 * 0.05 / 1e4 in the units of the
    # constraint, and the statistics refer to the original residuals.
    np.testing.assert_allclose(scaled.multipliers.equality, 1e-5, atol=1e-9)
    assert scaled.stats.infeasibility <= 1e-6
    assert scaled.stats.stationarity <= 1e-5
    # Without scaling the solve may fail, but a success has the same answer.
    if plain.result == alx.RESULTS.successful:
        np.testing.assert_allclose(plain.value, scaled.value, atol=1e-5)


def test_given_multipliers_are_in_original_units():
    multipliers = alx.Constraints(equality=jnp.asarray(1e-5), inequality=jnp.zeros(3))
    solution = alx.minimise(
        badly_scaled_projection,
        solver(alx.GradientScaling()),
        jnp.full(3, 1 / 3),
        POINT,
        multipliers=multipliers,
    )
    np.testing.assert_allclose(solution.value, [0.55, 0.45, 0.0], atol=1e-6)
    np.testing.assert_allclose(solution.multipliers.equality, 1e-5, atol=1e-9)


def test_derivatives_do_not_depend_on_the_scaling():
    def project(point, scaling):
        return alx.minimise(
            badly_scaled_projection,
            solver(scaling),
            jnp.full(3, 1 / 3),
            point,
            adjoint=optx.ImplicitAdjoint(),
        ).value

    jacobian = jax.jacfwd(lambda point: project(point, alx.GradientScaling()))(POINT)
    # On the active set {0, 1}, the projection subtracts the mean excess.
    expected = np.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(jacobian, expected, atol=1e-5)
