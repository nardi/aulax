import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest

import aulax as alx


SOLVER = alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-10, atol=1e-10))
ADJOINTS = {
    "implicit_dense": optx.ImplicitAdjoint(),
    "implicit_gmres": optx.ImplicitAdjoint(
        linear_solver=lx.GMRES(rtol=1e-10, atol=1e-10)
    ),
    "unrolled": optx.RecursiveCheckpointAdjoint(),
}


def simplex_projection(y, args, state):
    """Project `args` onto the probability simplex."""
    return alx.Evaluation(
        objective=jnp.sum((y - args) ** 2),
        equality=jnp.sum(y) - 1.0,
        inequality=-y,
    )


def project(point, adjoint):
    return alx.minimise(
        simplex_projection, SOLVER, jnp.full(3, 1 / 3), point, adjoint=adjoint
    ).value


POINT = jnp.array([0.6, 0.5, -0.4])
# On the active set {0, 1}, the projection subtracts the mean excess, so its
# Jacobian there is the identity minus 1/2 in every entry.
EXPECTED_JACOBIAN = np.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])


@pytest.mark.parametrize("adjoint_name", ["implicit_dense", "implicit_gmres"])
def test_implicit_jacobian_of_projection(adjoint_name):
    adjoint = ADJOINTS[adjoint_name]
    np.testing.assert_allclose(project(POINT, adjoint), [0.55, 0.45, 0.0], atol=1e-6)
    reverse = jax.jacrev(project)(POINT, adjoint)
    forward = jax.jacfwd(project)(POINT, adjoint)
    np.testing.assert_allclose(reverse, EXPECTED_JACOBIAN, atol=1e-6)
    np.testing.assert_allclose(forward, EXPECTED_JACOBIAN, atol=1e-6)


def test_unrolled_gradient_matches_implicit():
    def loss(point, adjoint):
        return jnp.sum(project(point, adjoint) * jnp.array([1.0, 2.0, 3.0]))

    implicit = jax.grad(loss)(POINT, ADJOINTS["implicit_dense"])
    unrolled = jax.grad(loss)(POINT, ADJOINTS["unrolled"])
    np.testing.assert_allclose(
        implicit, EXPECTED_JACOBIAN.T @ [1.0, 2.0, 3.0], atol=1e-6
    )
    np.testing.assert_allclose(unrolled, implicit, atol=1e-4)


def test_gradient_through_closed_over_array():
    def loss(point):
        def closed_projection(y, args, state):
            return simplex_projection(y, point, state)

        value = alx.minimise(closed_projection, SOLVER, jnp.full(3, 1 / 3)).value
        return value[0]

    np.testing.assert_allclose(jax.grad(loss)(POINT), EXPECTED_JACOBIAN[0], atol=1e-6)


def test_gradient_of_multipliers_matches_finite_differences():
    def equality_multiplier(point):
        solution = alx.minimise(simplex_projection, SOLVER, jnp.full(3, 1 / 3), point)
        return solution.multipliers.equality

    gradient = jax.grad(equality_multiplier)(POINT)
    step = 1e-5
    finite_differences = [
        (
            equality_multiplier(POINT.at[i].add(step))
            - equality_multiplier(POINT.at[i].add(-step))
        )
        / (2 * step)
        for i in range(3)
    ]
    np.testing.assert_allclose(gradient, finite_differences, atol=1e-4)


def test_vmap_of_gradient():
    points = jnp.array([[0.6, 0.5, -0.4], [0.2, 0.3, 0.5], [2.0, 0.0, 0.0]])
    gradients = jax.vmap(
        jax.grad(lambda point: project(point, ADJOINTS["implicit_dense"])[0])
    )(points)
    np.testing.assert_allclose(gradients[0], EXPECTED_JACOBIAN[0], atol=1e-6)
    # The second point already lies on the simplex, and all its constraints are
    # inactive, so the projection is the identity minus the mean on all three.
    np.testing.assert_allclose(gradients[1], [2 / 3, -1 / 3, -1 / 3], atol=1e-6)
    # The third point projects onto a vertex, where the result is locally fixed.
    np.testing.assert_allclose(gradients[2], [0.0, 0.0, 0.0], atol=1e-6)
