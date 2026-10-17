import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

import aulax as alx
from aulax._sparse_newton import _import_sparse_libraries


def test_missing_libraries_raise_an_informative_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "asdex", None)
    with pytest.raises(ImportError, match=r"aulax\[sparse\]"):
        _import_sparse_libraries()


requires_sparse_libraries = pytest.mark.skipif(
    importlib.util.find_spec("asdex") is None
    or importlib.util.find_spec("splineax") is None,
    reason="Install the `sparse` dependency group.",
)


def chain(y, args):
    """Return a function whose Hessian is tridiagonal, with the aux output unused."""
    value = (
        jnp.sum((y[1:] - y[:-1]) ** 2) + 0.01 * jnp.sum(y**4) + jnp.sum((y - args) ** 2)
    )
    return value, None


@requires_sparse_libraries
def test_minimises_a_banded_function_like_the_dense_method():
    size = 300
    target = jnp.sin(jnp.linspace(0.0, 3.0, size))
    y0 = jnp.zeros(size)
    sparse = optx.minimise(
        chain,
        alx.SparseNewton(rtol=1e-10, atol=1e-10),
        y0,
        target,
        has_aux=True,
        max_steps=200,
    )
    dense = optx.minimise(
        chain,
        alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10),
        y0,
        target,
        has_aux=True,
        max_steps=200,
    )
    np.testing.assert_allclose(sparse.value, dense.value, atol=1e-7)
    assert (
        float(jnp.max(jnp.abs(jax.grad(lambda y: chain(y, target)[0])(sparse.value))))
        < 1e-6
    )


def pendulum_like(y, args, state):
    """A chain of equality constraints and bounds with a banded Hessian."""
    return alx.Evaluation(
        objective=jnp.sum((y - 0.3) ** 2) + 0.1 * jnp.sum(jnp.diff(y) ** 2),
        equality=y[1:] - 0.5 * y[:-1] - 0.1 * jnp.sin(y[1:]),
        inequality=jnp.concatenate([-y - 1.0, y - 2.0]),
    )


@requires_sparse_libraries
def test_augmented_lagrangian_with_sparse_inner_solves():
    y0 = jnp.zeros(60)
    sparse_solver = alx.AugmentedLagrangian(
        inner=alx.SparseNewton(rtol=1e-10, atol=1e-10), stationarity_tolerance=1e-6
    )
    dense_solver = alx.AugmentedLagrangian(
        inner=alx.NewtonTrustRegion(rtol=1e-10, atol=1e-10), stationarity_tolerance=1e-6
    )
    sparse = alx.minimise(pendulum_like, sparse_solver, y0, max_steps=3000)
    dense = alx.minimise(pendulum_like, dense_solver, y0, max_steps=3000)
    np.testing.assert_allclose(sparse.value, dense.value, atol=1e-5)
    assert sparse.stats.infeasibility <= 1e-6
