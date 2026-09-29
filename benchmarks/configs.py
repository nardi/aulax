"""Named solver configurations that the benchmark scripts can run."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import scipy.optimize
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

import aulax as alx

from .metrics import classify_outcome, constraint_violation, flat_leaves
from .problems import BenchmarkProblem
from .records import SolveRecord


class Configuration(Protocol):
    """A solver setup that solves one benchmark problem and records the result."""

    # Read-only properties, so that frozen dataclasses satisfy the protocol.
    @property
    def name(self) -> str:
        """The name the scripts and result files use."""
        ...

    @property
    def description(self) -> str:
        """One sentence on what the configuration changes."""
        ...

    def solve(self, problem: BenchmarkProblem, repeats: int) -> SolveRecord:
        """Solve `problem`, timing the best of `repeats` runs, and return a record."""
        ...


def _as_jax(tree: PyTree) -> PyTree:
    """Return `tree` with every leaf converted to a JAX array."""
    return jax.tree.map(jnp.asarray, tree)


def _problem_sizes(problem: BenchmarkProblem) -> tuple[int, int, int]:
    """Return the number of variables, equality and inequality residuals."""
    evaluation = problem.problem(_as_jax(problem.y0), problem.args, None)
    return (
        flat_leaves(problem.y0).size,
        flat_leaves(evaluation.equality).size,
        flat_leaves(evaluation.inequality).size,
    )


def _solution_error(problem: BenchmarkProblem, value: PyTree) -> float | None:
    """Return the largest difference from the known minimizer, if there is one."""
    if problem.solution is None:
        return None
    difference = flat_leaves(value) - flat_leaves(problem.solution)
    return float(np.max(np.abs(difference)))


def _error_record(problem: BenchmarkProblem, error: Exception) -> SolveRecord:
    """Return the record of a solve that raised `error`."""
    variables, equality, inequality = _problem_sizes(problem)
    return SolveRecord(
        problem=problem.name,
        variables=variables,
        equality_constraints=equality,
        inequality_constraints=inequality,
        outcome="error",
        message=f"{type(error).__name__}: {error}",
        known_objective=problem.objective,
    )


def _best_time(function: Callable[[], object], repeats: int) -> float:
    """Return the fastest of `repeats` calls of `function`, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


@dataclass(frozen=True)
class AulaxConfiguration:
    """An aulax solver, run with `alx.minimise` under `jax.jit`."""

    name: str
    """The name the scripts and result files use."""
    description: str
    """One sentence on what the configuration changes."""
    solver: alx.AugmentedLagrangian
    """The solver to run."""
    max_steps: int = 20_000
    """The step limit, high enough that it only stops solves that are stuck."""

    def solve(self, problem: BenchmarkProblem, repeats: int) -> SolveRecord:
        """Solve `problem` and count the evaluations with a debug callback."""
        try:
            return self._solve(problem, repeats)
        except Exception as error:
            return _error_record(problem, error)

    def _solve(self, problem: BenchmarkProblem, repeats: int) -> SolveRecord:
        """Solve `problem`, letting exceptions through."""
        evaluation_count = 0

        def count_evaluation() -> None:
            nonlocal evaluation_count
            evaluation_count += 1

        def counted_problem(y, args, state):
            jax.debug.callback(count_evaluation)
            return problem.problem(y, args, state)

        @jax.jit
        def solve(y0, args):
            return alx.minimise(
                counted_problem,
                self.solver,
                y0,
                args,
                max_steps=self.max_steps,
                throw=False,
            )

        y0 = _as_jax(problem.y0)
        start = time.perf_counter()
        solution = jax.block_until_ready(solve(y0, problem.args))
        compile_seconds = time.perf_counter() - start
        jax.effects_barrier()
        evaluations = evaluation_count
        seconds = _best_time(
            lambda: jax.block_until_ready(solve(y0, problem.args)), repeats
        )

        value = jax.tree.map(np.asarray, solution.value)
        evaluation = problem.problem(solution.value, problem.args, None)
        objective = float(evaluation.objective)
        infeasibility = constraint_violation(evaluation)
        variables, equality, inequality = _problem_sizes(problem)
        converged = bool(solution.result == alx.RESULTS.successful)
        return SolveRecord(
            problem=problem.name,
            variables=variables,
            equality_constraints=equality,
            inequality_constraints=inequality,
            outcome=classify_outcome(
                converged, infeasibility, objective, problem.objective
            ),
            message=alx.RESULTS[solution.result] or "successful",
            evaluations=evaluations,
            # Every evaluation computes derivatives, except the one at `y0` that
            # sets up the initial penalty.
            derivative_evaluations=evaluations - 1,
            outer_steps=int(solution.stats.num_outer_steps),
            objective=objective,
            known_objective=problem.objective,
            solution_error=_solution_error(problem, value),
            infeasibility=infeasibility,
            stationarity=float(solution.stats.stationarity),
            seconds=seconds,
            compile_seconds=compile_seconds,
        )


class _CachedDerivatives:
    """The values and derivatives of a problem as flat NumPy arrays, for SciPy.

    SciPy calls the objective, the constraints and their derivatives through
    separate functions, often at the same point. This class evaluates the problem
    once per point for values, and once per point for derivatives, and counts
    those evaluations. That matches what a user with one problem function would
    pay, which keeps the counts comparable with aulax.
    """

    def __init__(self, problem: BenchmarkProblem):
        flat_y0, unravel = ravel_pytree(_as_jax(problem.y0))
        self.flat_y0 = np.asarray(flat_y0)
        """The starting point as one flat array."""
        self.unravel = unravel
        """The function that turns a flat array back into the shape of `y0`."""

        def flat_values(flat_y):
            evaluation = problem.problem(unravel(flat_y), problem.args, None)
            return (
                evaluation.objective,
                _flat(evaluation.equality),
                _flat(evaluation.inequality),
            )

        self._values = jax.jit(flat_values)
        self._derivatives = jax.jit(
            lambda y: (flat_values(y), jax.jacfwd(flat_values)(y))
        )
        self._value_point: bytes | None = None
        self._value_cache = None
        self._derivative_point: bytes | None = None
        self._derivative_cache = None
        self.evaluations = 0
        """The number of problem evaluations, with or without derivatives."""
        self.derivative_evaluations = 0
        """How many of the evaluations also computed derivatives."""

    def values(self, x: np.ndarray):
        """Return the objective, equality and inequality residuals at `x`."""
        point = np.asarray(x, dtype=float).tobytes()
        if point == self._derivative_point and self._derivative_cache is not None:
            return self._derivative_cache[0]
        if point != self._value_point or self._value_cache is None:
            self._value_point = point
            self._value_cache = jax.tree.map(np.asarray, self._values(x))
            self.evaluations += 1
        return self._value_cache

    def derivatives(self, x: np.ndarray):
        """Return the Jacobians of the objective and the residuals at `x`."""
        point = np.asarray(x, dtype=float).tobytes()
        if point != self._derivative_point or self._derivative_cache is None:
            self._derivative_point = point
            self._derivative_cache = jax.tree.map(np.asarray, self._derivatives(x))
            self.evaluations += 1
            self.derivative_evaluations += 1
        return self._derivative_cache[1]


def _flat(tree) -> jax.Array:
    """Return the leaves of `tree` as one flat array, empty if there are none."""
    leaves = [jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)]
    return jnp.concatenate(leaves) if leaves else jnp.zeros(0)


@dataclass(frozen=True)
class ScipyConfiguration:
    """A method of `scipy.optimize.minimize`, with derivatives from JAX."""

    name: str
    """The name the scripts and result files use."""
    description: str
    """One sentence on what the configuration changes."""
    method: str
    """The `method` argument of `scipy.optimize.minimize`."""
    tolerance: float = 1e-10
    """The `tol` argument of `scipy.optimize.minimize`."""
    max_iterations: int = 5000
    """The iteration limit of the method."""

    def solve(self, problem: BenchmarkProblem, repeats: int) -> SolveRecord:
        """Solve `problem` and count the distinct evaluations SciPy asked for."""
        try:
            return self._solve(problem, repeats)
        except Exception as error:
            return _error_record(problem, error)

    def _run(self, problem: BenchmarkProblem, cache: _CachedDerivatives):
        """Run SciPy once with the evaluations from `cache`."""
        constraints = []
        _, equality, inequality = cache.values(cache.flat_y0)
        if equality.size:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: cache.values(x)[1],
                    "jac": lambda x: cache.derivatives(x)[1],
                }
            )
        if inequality.size:
            # SciPy writes inequality constraints as c(x) >= 0.
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: -cache.values(x)[2],
                    "jac": lambda x: -cache.derivatives(x)[2],
                }
            )
        return scipy.optimize.minimize(
            lambda x: float(cache.values(x)[0]),
            cache.flat_y0,
            jac=lambda x: cache.derivatives(x)[0],
            method=self.method,
            constraints=constraints,
            tol=self.tolerance,
            options={"maxiter": self.max_iterations},
        )

    def _solve(self, problem: BenchmarkProblem, repeats: int) -> SolveRecord:
        """Solve `problem`, letting exceptions through."""
        cache = _CachedDerivatives(problem)
        start = time.perf_counter()
        result = self._run(problem, cache)
        compile_seconds = time.perf_counter() - start
        seconds = _best_time(
            lambda: self._run(problem, _CachedDerivatives(problem)), repeats
        )
        value = cache.unravel(jnp.asarray(result.x))
        evaluation = problem.problem(value, problem.args, None)
        objective = float(evaluation.objective)
        infeasibility = constraint_violation(evaluation)
        variables, equality, inequality = _problem_sizes(problem)
        return SolveRecord(
            problem=problem.name,
            variables=variables,
            equality_constraints=equality,
            inequality_constraints=inequality,
            outcome=classify_outcome(
                bool(result.success), infeasibility, objective, problem.objective
            ),
            message=str(result.message),
            evaluations=cache.evaluations,
            derivative_evaluations=cache.derivative_evaluations,
            objective=objective,
            known_objective=problem.objective,
            solution_error=_solution_error(problem, value),
            infeasibility=infeasibility,
            seconds=seconds,
            compile_seconds=compile_seconds,
        )


_CONFIGURATION_LIST: list[Configuration] = [
    AulaxConfiguration(
        name="baseline",
        description="The default solver with BFGS inner solves.",
        solver=alx.AugmentedLagrangian(inner=optx.BFGS(rtol=1e-8, atol=1e-8)),
    ),
    ScipyConfiguration(
        name="scipy_slsqp",
        description="SciPy's SLSQP.",
        method="SLSQP",
    ),
    ScipyConfiguration(
        name="scipy_trust_constr",
        description="SciPy's trust-constr with quasi-Newton Hessians.",
        method="trust-constr",
    ),
]
"""The configurations, in the order the scripts list them."""

CONFIGURATIONS: dict[str, Configuration] = {
    configuration.name: configuration for configuration in _CONFIGURATION_LIST
}
"""The configurations by name."""
