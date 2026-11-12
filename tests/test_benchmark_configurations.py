"""Tests for the benchmark configurations that pass the bounds to the solver."""

from importlib.util import find_spec

import pytest

import aulax as alx


requires_optax = pytest.mark.skipif(
    find_spec("optax") is None,
    reason="`benchmarks.configs` imports `optax` from the `bench` dependency group.",
)


@requires_optax
@pytest.mark.parametrize("name", ["newton_bounds", "lbfgs_bounds"])
def test_native_bounds_configuration_is_registered(name: str):
    from benchmarks.configs import CONFIGURATIONS, AulaxConfiguration

    configuration = CONFIGURATIONS[name]
    assert isinstance(configuration, AulaxConfiguration)
    assert configuration.name == name
    assert configuration.native_bounds


@requires_optax
@pytest.mark.parametrize("name", ["newton_bounds", "lbfgs_bounds"])
def test_native_bounds_configuration_has_a_one_sentence_description(name: str):
    from benchmarks.configs import CONFIGURATIONS

    description = CONFIGURATIONS[name].description
    assert description.endswith(".")
    assert description.count(".") == 1


@requires_optax
def test_newton_bounds_uses_the_default_newton_minimiser():
    from benchmarks.configs import CONFIGURATIONS, AulaxConfiguration

    configuration = CONFIGURATIONS["newton_bounds"]
    assert isinstance(configuration, AulaxConfiguration)
    assert isinstance(configuration.solver.inner, alx.Newton)
    assert configuration.solver.scaling == alx.AugmentedLagrangian().scaling


@requires_optax
def test_lbfgs_bounds_uses_the_default_lbfgs_minimiser():
    from benchmarks.configs import CONFIGURATIONS, AulaxConfiguration

    configuration = CONFIGURATIONS["lbfgs_bounds"]
    assert isinstance(configuration, AulaxConfiguration)
    assert type(configuration.solver.inner).__name__ == "LBFGS"
    assert configuration.solver.scaling == alx.AugmentedLagrangian().scaling


@requires_optax
@pytest.mark.parametrize("name", ["baseline", "newton", "scipy_slsqp", "ipopt"])
def test_existing_configurations_do_not_pass_bounds_natively(name: str):
    from benchmarks.configs import CONFIGURATIONS, AulaxConfiguration

    configuration = CONFIGURATIONS[name]
    if isinstance(configuration, AulaxConfiguration):
        assert not configuration.native_bounds
