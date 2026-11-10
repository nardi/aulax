import jax.numpy as jnp

import aulax as alx


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
