from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from ._lagrangian import (
    augmented_lagrangian,
    kkt_infeasibility,
    lagrangian,
    penalty_infeasibility,
)
from ._misc import tree_any, tree_full_like, tree_max_norm, tree_where
from ._problem import Args, Aux, Constraints, Evaluation, ProblemFunction, State, Y
from ._solution import RESULTS, Solution, Statistics
from ._solver import AugmentedLagrangian


class _InnerArgs(eqx.Module, Generic[Args, State]):
    """The `args` of the augmented Lagrangian that the inner solver minimizes."""

    args: Args
    """The user's `args`."""
    state: State
    """The problem state to pass to the next evaluation."""
    multipliers: Constraints
    """The safeguarded multipliers of the current augmented Lagrangian."""
    penalty: Constraints
    """The penalty of the current augmented Lagrangian."""


InnerFunction = Callable[[Y, _InnerArgs], tuple[Scalar, Evaluation]]
"""The augmented Lagrangian as an optimistix function, with the evaluation as aux."""


def _inner_function(problem: ProblemFunction) -> InnerFunction:
    """Wrap `problem` as the augmented Lagrangian in the optimistix signature."""

    def augmented_lagrangian_function(
        y: Y, inner_args: _InnerArgs
    ) -> tuple[Scalar, Evaluation]:
        state = jax.lax.stop_gradient(inner_args.state)
        evaluation = problem(y, inner_args.args, state)
        value = augmented_lagrangian(
            evaluation, inner_args.multipliers, inner_args.penalty
        )
        return value, evaluation

    return augmented_lagrangian_function


class _LoopState(eqx.Module):
    """The carry of the solver loop."""

    y: PyTree[Array]
    """The current point of the inner solve."""
    inner_state: PyTree[Array]
    """The array part of the inner solver state."""
    evaluation: Evaluation
    """The latest evaluation the inner solver returned.

    It belongs to the point that went into the last inner step, and to the inner
    solution right after an outer update.
    """
    multipliers: Constraints
    """The safeguarded multipliers of the current augmented Lagrangian."""
    multiplier_estimates: Constraints
    """The multiplier estimates from the last outer update."""
    penalty: Constraints
    """The penalty of the current augmented Lagrangian."""
    previous_infeasibility: Constraints
    """The elementwise infeasibility after the previous inner solve."""
    inner_tolerance: Scalar
    """The tolerance of the current inner solve."""
    inner_steps: Int[Array, ""]
    """The number of steps in the current inner solve."""
    update_pending: Bool[Array, ""]
    """Whether the inner solve has finished and the next iteration updates."""
    num_steps: Int[Array, ""]
    """The number of loop iterations so far."""
    num_outer_steps: Int[Array, ""]
    """The number of outer updates so far."""
    terminate: Bool[Array, ""]
    """Whether the solve has converged."""
    result: RESULTS
    """The outcome of the solve so far."""


def _as_inexact_array(leaf: Any) -> Array:
    """Convert `leaf` to an array with a floating-point dtype."""
    array = jnp.asarray(leaf)
    if jnp.issubdtype(array.dtype, jnp.inexact):
        return array
    return array.astype(float)


def _python_float_as_array(leaf: Any) -> Any:
    """Convert a Python float or complex number to an array, and keep the rest.

    The adjoint turns floating-point leaves into arrays on the way in, so the
    problem has to see them as arrays from the start.
    """
    if isinstance(leaf, float | complex):
        return jnp.asarray(leaf)
    return leaf


def _tree_all_finite(tree: PyTree[Array]) -> Bool[Array, ""]:
    """Return whether every element of `tree` is finite."""
    leaves = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.array(True)


class _SolveOutputs(eqx.Module):
    """The outputs of a solve that are not differentiated."""

    evaluation: Evaluation
    """The evaluation at the solution, with its state and aux output."""
    active: PyTree[Bool[Array, "..."]]
    """Which inequality constraints are active at the solution."""
    result: RESULTS
    """The outcome of the solve."""
    stats: Statistics
    """Statistics about the solve."""


def _solve(inputs: tuple) -> tuple[tuple[PyTree[Array], Constraints], _SolveOutputs]:
    """Run the solver loop and return the minimizer and multipliers.

    `inputs` holds the problem, the solver, `y0`, `args`, the initial state and
    multipliers, the inner solver options, the step limit and the while loop to
    run. It is a single tuple so that an optimistix adjoint can differentiate
    through it.
    """
    problem, solver, y0, args, state, multipliers, options, max_steps, while_loop = (
        inputs
    )
    tags = frozenset()

    def objective_with_evaluation(y: Y) -> tuple[Scalar, Evaluation]:
        evaluation = problem(y, args, state)
        return evaluation.objective, evaluation

    # The first evaluation also computes the objective gradient, whose size sets
    # the scale of the gradient-based stopping tests.
    objective_gradient, evaluation0 = jax.grad(objective_with_evaluation, has_aux=True)(
        y0
    )
    gradient_scale = jax.lax.stop_gradient(
        jnp.maximum(1.0, tree_max_norm(objective_gradient))
    )
    objective = jnp.asarray(evaluation0.objective)

    if multipliers is None:
        multipliers = Constraints(
            equality=tree_full_like(evaluation0.equality, 0.0),
            inequality=tree_full_like(evaluation0.inequality, 0.0),
        )
    safeguarded_multipliers = solver.multiplier_update.safeguard_multipliers(
        multipliers, evaluation0
    )
    penalty = solver.penalty_update.initial_penalty(evaluation0)
    inner_tolerance = jnp.asarray(
        solver.tolerance_schedule.initial_tolerance(), dtype=objective.dtype
    )

    inner_function = _inner_function(problem)
    inner_args0 = _InnerArgs(args, evaluation0.state, safeguarded_multipliers, penalty)
    f_struct, aux_struct = jax.eval_shape(inner_function, y0, inner_args0)
    inner_state0 = solver.inner.init(
        inner_function, y0, inner_args0, options, f_struct, aux_struct, tags
    )
    # A loop carry holds arrays only, so the static part of the inner state stays
    # outside and is combined with the array part inside the loop.
    dynamic_inner_state0, static_inner_state = eqx.partition(inner_state0, eqx.is_array)

    def reinitialize_inner_state(y: Y, inner_args: _InnerArgs) -> PyTree[Array]:
        """Return the array part of a fresh inner solver state at `y`."""
        new_state = solver.inner.init(
            inner_function, y, inner_args, options, f_struct, aux_struct, tags
        )
        dynamic_state, _ = eqx.partition(new_state, eqx.is_array)
        return dynamic_state

    def inner_step(loop: _LoopState) -> tuple[_LoopState, PyTree]:
        """Take one inner step, and mark the inner solve finished if it has.

        Returns the new loop state and the full new inner solver state, whose
        accepted evaluation and gradient the outer update may use.
        """
        inner = solver.inner_at_tolerance(loop.inner_tolerance)
        inner_args = _InnerArgs(
            args, loop.evaluation.state, loop.multipliers, loop.penalty
        )
        inner_state = eqx.combine(loop.inner_state, static_inner_state)
        new_y, new_inner_state, evaluation = inner.step(
            inner_function, loop.y, inner_args, options, inner_state, tags
        )
        inner_terminate, inner_result = inner.terminate(
            inner_function, new_y, inner_args, options, new_inner_state, tags
        )
        new_dynamic_inner_state, _ = eqx.partition(new_inner_state, eqx.is_array)
        inner_steps = loop.inner_steps + 1
        inner_converged = solver.inner_stopping.is_inner_converged(
            solver.inner_state_access.accepted_gradient(new_inner_state),
            loop.inner_tolerance,
            gradient_scale,
            inner_terminate,
        )
        result = RESULTS.where(
            _tree_all_finite(new_y),
            RESULTS.promote(inner_result),
            RESULTS.nonfinite,  # ty: ignore[invalid-argument-type]
        )
        stepped = eqx.tree_at(
            lambda loop: (
                loop.y,
                loop.inner_state,
                loop.evaluation,
                loop.inner_steps,
                loop.update_pending,
                loop.result,
            ),
            loop,
            (
                new_y,
                new_dynamic_inner_state,
                evaluation,
                inner_steps,
                inner_converged | (inner_steps >= solver.inner_max_steps),
                result,
            ),
        )
        return stepped, new_inner_state

    def outer_update(
        loop: _LoopState, evaluation: Evaluation, gradient: PyTree[Array] | None
    ) -> _LoopState:
        """Update multipliers and penalty from the inner solution, and restart.

        `evaluation` belongs to `loop.y`, the inner solution. `gradient` is the
        gradient of the augmented Lagrangian there, if the inner solver provides
        it. It equals the gradient of the Lagrangian with the new multiplier
        estimates, so its size measures stationarity.
        """
        estimates = solver.multiplier_update.update_multipliers(
            loop.multipliers, loop.penalty, evaluation
        )
        infeasibility = penalty_infeasibility(
            evaluation, loop.multipliers, loop.penalty
        )
        new_penalty, penalty_stalled = solver.penalty_update.update_penalty(
            loop.penalty, infeasibility, loop.previous_infeasibility
        )
        new_multipliers = solver.multiplier_update.safeguard_multipliers(
            estimates, evaluation
        )
        if gradient is None:
            stationary = jnp.array(True)
        else:
            stationary = (
                tree_max_norm(gradient)
                <= solver.stationarity_tolerance * gradient_scale
            )
        converged = (
            solver.is_final_tolerance(loop.inner_tolerance)
            & (
                kkt_infeasibility(evaluation, estimates)
                <= solver.feasibility_tolerance_for(objective.dtype)
            )
            & stationary
        )
        infeasible = penalty_stalled & jnp.invert(converged)
        new_tolerance = jnp.maximum(
            solver.tolerance_schedule.update_tolerance(loop.inner_tolerance),
            solver.final_inner_tolerance(),
        )
        new_inner_args = _InnerArgs(
            args, evaluation.state, new_multipliers, new_penalty
        )
        fresh_inner_state = eqx.combine(
            reinitialize_inner_state(loop.y, new_inner_args), static_inner_state
        )
        transferred_inner_state = solver.inner_state_access.transfer_state(
            eqx.combine(loop.inner_state, static_inner_state),
            fresh_inner_state,
            tree_any(
                jax.tree.map(jnp.greater, new_penalty, loop.penalty),
            ),
        )
        new_dynamic_inner_state, _ = eqx.partition(
            transferred_inner_state, eqx.is_array
        )
        return _LoopState(
            y=loop.y,
            inner_state=new_dynamic_inner_state,
            evaluation=evaluation,
            multipliers=new_multipliers,
            multiplier_estimates=estimates,
            penalty=new_penalty,
            previous_infeasibility=infeasibility,
            inner_tolerance=new_tolerance.astype(loop.inner_tolerance.dtype),
            inner_steps=jnp.array(0),
            update_pending=jnp.array(False),
            num_steps=loop.num_steps,
            num_outer_steps=loop.num_outer_steps + 1,
            terminate=converged | infeasible,
            result=RESULTS.where(infeasible, RESULTS.infeasible, loop.result),
        )

    def body(loop: _LoopState) -> _LoopState:
        """Run one loop iteration, which always calls the inner `step` once.

        When the inner solver state holds the evaluation at its accepted point,
        the multipliers update in the same iteration that the inner solve ends.
        Otherwise the loop only has the evaluation of the point that went into
        `step`. It then steps once more from the inner solution to obtain its
        evaluation, updates the multipliers from it, and discards that step.
        """
        stepped, new_inner_state = inner_step(loop)
        accepted = solver.inner_state_access.accepted_evaluation(new_inner_state)
        if accepted is None:
            updated = outer_update(loop, stepped.evaluation, None)
            new_loop = tree_where(loop.update_pending, updated, stepped)
        else:
            gradient = solver.inner_state_access.accepted_gradient(new_inner_state)
            stepped = eqx.tree_at(lambda loop: loop.evaluation, stepped, accepted)
            updated = outer_update(stepped, accepted, gradient)
            new_loop = tree_where(stepped.update_pending, updated, stepped)
        return eqx.tree_at(lambda loop: loop.num_steps, new_loop, loop.num_steps + 1)

    def keep_going(loop: _LoopState) -> Bool[Array, ""]:
        """Return whether the loop continues."""
        return (
            jnp.invert(loop.terminate)
            & (loop.result == RESULTS.successful)
            & (loop.num_steps < max_steps)
        )

    loop0 = _LoopState(
        y=y0,
        inner_state=dynamic_inner_state0,
        evaluation=evaluation0,
        multipliers=safeguarded_multipliers,
        multiplier_estimates=multipliers,
        penalty=penalty,
        previous_infeasibility=tree_full_like(
            Constraints(evaluation0.equality, evaluation0.inequality), jnp.inf
        ),
        inner_tolerance=inner_tolerance,
        inner_steps=jnp.array(0),
        update_pending=jnp.array(False),
        num_steps=jnp.array(0),
        num_outer_steps=jnp.array(0),
        terminate=jnp.array(False),
        result=RESULTS.successful,  # ty: ignore[invalid-argument-type]
    )
    loop = while_loop(keep_going, body, loop0, max_steps=max_steps)
    result = RESULTS.where(
        (loop.result == RESULTS.successful) & jnp.invert(loop.terminate),
        RESULTS.nonlinear_max_steps_reached,  # ty: ignore[invalid-argument-type]
        loop.result,
    )

    def solution_lagrangian(y: Y) -> Scalar:
        evaluation = problem(y, args, jax.lax.stop_gradient(loop.evaluation.state))
        return lagrangian(evaluation, loop.multiplier_estimates)

    stationarity = tree_max_norm(jax.grad(solution_lagrangian)(loop.y))
    # An inequality constraint counts as active when its multiplier is positive.
    # A constraint that holds with equality but has a zero multiplier is treated
    # as inactive, which is one of the valid derivatives at such a point.
    active = jax.tree.map(
        lambda multiplier: multiplier > 0.0, loop.multiplier_estimates.inequality
    )
    outputs = _SolveOutputs(
        evaluation=loop.evaluation,
        active=active,
        result=result,
        stats=Statistics(
            num_steps=loop.num_steps,
            num_outer_steps=loop.num_outer_steps,
            penalty=loop.penalty,
            infeasibility=kkt_infeasibility(loop.evaluation, loop.multiplier_estimates),
            stationarity=stationarity,
        ),
    )
    return (loop.y, loop.multiplier_estimates), outputs


def _kkt_residual(
    root: tuple[PyTree[Array], Constraints], outputs: _SolveOutputs, inputs: tuple
) -> tuple[PyTree[Array], Constraints]:
    r"""Return the KKT residual whose root is the solution, for implicit derivatives.

    The residual is $\nabla_y L$, then $h$, then $g_j$ for an active inequality
    and $-\mu_j$ for an inactive one. The active set is fixed at the solution, so
    the residual keeps its shape whichever constraints are active. The gradient
    only includes the multipliers of active constraints, which makes the Jacobian
    with respect to `root` symmetric.
    """
    y, multipliers = root
    problem, _, _, args, *_ = inputs
    state = jax.lax.stop_gradient(outputs.evaluation.state)
    active_multipliers = Constraints(
        equality=multipliers.equality,
        inequality=jax.tree.map(
            lambda multiplier, active: jnp.where(active, multiplier, 0.0),
            multipliers.inequality,
            outputs.active,
        ),
    )

    def solution_lagrangian(y: PyTree[Array]) -> tuple[Scalar, Evaluation]:
        evaluation = problem(y, args, state)
        return lagrangian(evaluation, active_multipliers), evaluation

    gradient, evaluation = jax.grad(solution_lagrangian, has_aux=True)(y)
    inequality_residual = jax.tree.map(
        lambda residual, multiplier, active: jnp.where(active, residual, -multiplier),
        evaluation.inequality,
        multipliers.inequality,
        outputs.active,
    )
    return gradient, Constraints(evaluation.equality, inequality_residual)


@eqx.filter_jit
def minimise(
    problem: ProblemFunction[Y, Args, State, Aux],
    solver: AugmentedLagrangian,
    y0: Y,
    # `None` is the default for problems without `args` or `state`, which a
    # generic parameter cannot express.
    args: Args = None,  # ty: ignore[invalid-parameter-default]
    state: State = None,  # ty: ignore[invalid-parameter-default]
    *,
    multipliers: Constraints | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int = 1024,
    adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(),
    throw: bool = True,
) -> Solution[Y, State, Aux]:
    """Minimize a function subject to equality and inequality constraints.

    Args:
        problem: The problem function, called as `problem(y, args, state)`. It
            returns an `Evaluation` with the objective, the constraint residuals,
            the next state and any aux output.
        solver: The solver, usually an `AugmentedLagrangian`.
        y0: The initial point, any pytree of arrays.
        args: Passed as the `args` of every call to `problem`.
        state: The state for the first call to `problem`. Later calls receive the
            `state` returned by the previous evaluation.
        multipliers: Initial multipliers, shaped like the constraints. `None`
            starts from zero.
        options: Runtime options passed to the inner solver.
        max_steps: The maximum number of inner steps, summed over all inner
            solves.
        adjoint: How to differentiate the solution. `optimistix.ImplicitAdjoint`
            applies the implicit function theorem to the KKT conditions, and
            `optimistix.RecursiveCheckpointAdjoint` differentiates through every
            step of the solve.
        throw: Whether to raise an error when the solve fails. If `False`, check
            `Solution.result` instead.

    Returns:
        A `Solution` with the minimizer, the multipliers and the final state.
        Derivatives flow through `value` and `multipliers` only.
    """
    y0 = jax.tree.map(_as_inexact_array, y0)
    args = jax.tree.map(_python_float_as_array, args)
    state = jax.tree.map(_python_float_as_array, state)
    options = {} if options is None else options
    # Closure conversion turns arrays that `problem` closes over into leaves of a
    # pytree, so derivatives with respect to them reach the adjoint.
    problem = eqx.filter_closure_convert(problem, y0, args, state)
    objective_struct = jax.eval_shape(problem, y0, args, state).objective
    if objective_struct.shape != () or not jnp.issubdtype(
        objective_struct.dtype, jnp.floating
    ):
        raise ValueError("The objective must be a floating-point scalar.")

    inputs = (problem, solver, y0, args, state, multipliers, options, max_steps)
    # The KKT residual has a symmetric Jacobian, see `_kkt_residual`.
    tags = frozenset({lx.symmetric_tag})
    (value, solution_multipliers), outputs = adjoint.apply(
        _solve, _kkt_residual, inputs, tags
    )
    solution = Solution(
        value=value,
        result=outputs.result,
        multipliers=solution_multipliers,
        state=outputs.evaluation.state,
        aux=outputs.evaluation.aux,
        stats=outputs.stats,
    )
    if throw:
        solution = outputs.result.error_if(
            solution, outputs.result != RESULTS.successful
        )
    return solution
