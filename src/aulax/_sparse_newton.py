import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from jax.experimental.sparse import BCOO
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from ._misc import tree_where


def _import_sparse_libraries():
    """Import `asdex` and `splineax`, which `SparseNewton` needs."""
    try:
        import asdex
        import splineax
    except ImportError as error:
        raise ImportError(
            "`SparseNewton` needs the `asdex` and `splineax` packages. Install "
            "them with `pip install aulax[sparse]`."
        ) from error
    return asdex, splineax


class _SparseInfo(eqx.Module):
    """The value, gradient and Hessian entries at the accepted point."""

    f: Scalar
    """The function value."""
    grad: PyTree[Float[Array, "..."]]
    """The gradient, shaped like `y`."""
    hessian_data: Float[Array, " nnz"]
    """The Hessian entries, in the order of the detected sparsity pattern."""


class _SparseState(eqx.Module):
    """The state of `SparseNewton` between steps."""

    first_step: Bool[Array, ""]
    """Whether the next evaluation is the first one, which is always accepted."""
    y_eval: PyTree[Float[Array, "..."]]
    """The trial point that the next step evaluates."""
    f_info: _SparseInfo
    """The function value, gradient and Hessian at the last accepted point."""
    aux: PyTree
    """The aux output at the last accepted point."""
    damping: Scalar
    """The Levenberg-Marquardt damping, added to the diagonal of the Hessian."""
    terminate: Bool[Array, ""]
    """Whether the last step moved less than the tolerances allow."""
    result: optx.RESULTS
    """The outcome of the solve so far."""


@dataclasses.dataclass(frozen=True)
class _Structure:
    """The sparsity of the Hessian, and of the Hessian plus a diagonal."""

    coloring: Any
    """The `asdex` coloring that materializes the Hessian."""
    rows: np.ndarray
    """Row indices of the Hessian entries."""
    cols: np.ndarray
    """Column indices of the Hessian entries."""
    union_rows: np.ndarray
    """Row indices of the entries of the Hessian plus a full diagonal."""
    union_cols: np.ndarray
    """Column indices of the entries of the Hessian plus a full diagonal."""
    hessian_positions: np.ndarray
    """Position of every Hessian entry among the union entries."""
    diagonal_positions: np.ndarray
    """Position of every diagonal entry among the union entries."""


def _detect_structure(fn: Callable, y: PyTree, args: PyTree) -> _Structure:
    """Detect the sparsity of the Hessian of `fn` with respect to `y`.

    The sparsity depends on how `fn` combines its inputs and not on their values,
    but the detection folds constants, so the sample arguments are ones to keep
    divisions by the penalty finite.
    """
    asdex, _ = _import_sparse_libraries()
    flat_y, unravel = ravel_pytree(y)
    sample_args = jax.tree.map(
        lambda leaf: (
            np.ones(jnp.shape(leaf), jnp.result_type(leaf))
            if isinstance(leaf, jax.Array)
            else leaf
        ),
        args,
    )

    def scalar_function(flat: Array, arguments: PyTree = sample_args) -> Scalar:
        return fn(unravel(flat), arguments)[0]

    coloring = asdex.hessian_coloring(
        scalar_function, np.zeros(flat_y.shape, flat_y.dtype)
    )
    size = flat_y.size
    rows = np.asarray(coloring.sparsity.rows, dtype=np.int64)
    cols = np.asarray(coloring.sparsity.cols, dtype=np.int64)
    diagonal = np.arange(size, dtype=np.int64)
    keys = np.concatenate([rows * size + cols, diagonal * size + diagonal])
    union, inverse = np.unique(keys, return_inverse=True)
    return _Structure(
        coloring=coloring,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        union_rows=(union // size).astype(np.int32),
        union_cols=(union % size).astype(np.int32),
        hessian_positions=inverse[: rows.size],
        diagonal_positions=inverse[rows.size :],
    )


class SparseNewton(optx.AbstractMinimiser):
    r"""Damped Newton's method with a sparse Hessian and a sparse direct solver.

    The sparsity of the Hessian is detected with `asdex`, which also materializes
    the Hessian with as many Hessian-vector products as the coloring of its
    sparsity graph needs. Each step solves $(H + \\sigma I) p = -g$ with a
    `splineax` solver, so the cost grows with the number of nonzeros and not with
    the square of the number of variables. The damping $\\sigma$ follows the
    Levenberg-Marquardt rule. It shrinks after a step that decreased the function
    as predicted, and it grows after a step that did not. A direction that is not
    a descent direction is replaced by the damped steepest descent step.

    It needs the optional packages `asdex` and `splineax` (`pip install
    aulax[sparse]`). The function must be traceable by `asdex`, which excludes
    callbacks. A dense row or column in the Hessian, such as a constraint that
    involves every variable, makes the coloring use as many colors as variables,
    and then `NewtonTrustRegion` is the better choice.
    """

    rtol: float
    """The relative tolerance of the stopping test."""
    atol: float
    """The absolute tolerance of the stopping test."""
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    """The norm in which the change of `y` is measured for the stopping test."""
    initial_damping: float = 1e-3
    """The damping of the first step."""
    accept_ratio: float = 1e-4
    """The smallest ratio of actual to predicted decrease that accepts a step."""
    linear_solver: Any = None
    """The `splineax` solver, or `None` for `splineax.AutoSparseLinearSolver`."""

    def init(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _SparseState:
        """Return a state whose first step evaluates the function at `y`."""
        flat_y, _ = ravel_pytree(y)
        structure = _detect_structure(fn, y, args)
        f_info = _SparseInfo(
            f=jnp.zeros((), f_struct.dtype),
            grad=jax.tree.map(jnp.zeros_like, y),
            hessian_data=jnp.zeros(structure.rows.size, flat_y.dtype),
        )
        return _SparseState(
            first_step=jnp.array(True),
            y_eval=y,
            f_info=f_info,
            aux=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), aux_struct
            ),
            damping=jnp.asarray(self.initial_damping, flat_y.dtype),
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
        )

    def step(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        state: _SparseState,
        tags: frozenset[object],
    ) -> tuple[PyTree[Float[Array, "..."]], _SparseState, PyTree]:
        """Evaluate the trial point, accept or reject it, and plan the next trial."""
        asdex, splineax = _import_sparse_libraries()
        solver = (
            splineax.AutoSparseLinearSolver()
            if self.linear_solver is None
            else self.linear_solver
        )
        flat_eval, unravel = ravel_pytree(state.y_eval)
        size = flat_eval.size
        structure = _detect_structure(fn, state.y_eval, args)
        pattern = jnp.stack(
            [jnp.asarray(structure.rows), jnp.asarray(structure.cols)], axis=1
        )

        (f_eval, aux_eval), g_eval = jax.value_and_grad(
            lambda flat: fn(unravel(flat), args), has_aux=True
        )(flat_eval)
        flat_y, _ = ravel_pytree(y)
        trial_step = flat_eval - flat_y
        old = state.f_info
        old_gradient, _ = ravel_pytree(old.grad)
        old_hessian = BCOO((old.hessian_data, pattern), shape=(size, size))
        predicted = -(
            old_gradient @ trial_step + 0.5 * trial_step @ (old_hessian @ trial_step)
        )
        ratio = (old.f - f_eval) / jnp.where(predicted > 0, predicted, jnp.inf)
        finite = jnp.isfinite(f_eval)
        accept = state.first_step | ((ratio > self.accept_ratio) & finite)

        # The Hessian is only needed at points that are accepted.
        hessian_function = asdex.hessian_from_coloring(
            lambda flat: fn(unravel(flat), args)[0], structure.coloring
        )
        hessian_data = jax.lax.cond(
            accept, lambda: hessian_function(flat_eval).data, lambda: old.hessian_data
        )
        evaluated = _SparseInfo(
            f=f_eval, grad=unravel(g_eval), hessian_data=hessian_data
        )
        f_info = tree_where(accept, evaluated, old)
        aux = tree_where(accept, aux_eval, state.aux)
        flat_new_y = jnp.where(accept, flat_eval, flat_y)

        damping = jnp.where(ratio > 0.75, state.damping / 3.0, state.damping)
        damping = jnp.where((ratio < 0.25) | jnp.invert(finite), damping * 4.0, damping)
        damping = jnp.where(state.first_step, state.damping, damping)

        gradient, _ = ravel_pytree(f_info.grad)
        shift = damping
        data = (
            jnp.zeros(structure.union_rows.shape, flat_eval.dtype)
            .at[jnp.asarray(structure.hessian_positions)]
            .add(f_info.hessian_data)
            .at[jnp.asarray(structure.diagonal_positions)]
            .add(shift)
        )
        union_pattern = jnp.stack(
            [jnp.asarray(structure.union_rows), jnp.asarray(structure.union_cols)],
            axis=1,
        )
        matrix = BCOO((data, union_pattern), shape=(size, size))
        solution = lx.linear_solve(
            splineax.BCOOLinearOperator(matrix), -gradient, solver, throw=False
        )
        is_descent = (gradient @ solution.value < 0) & jnp.all(
            jnp.isfinite(solution.value)
        )
        direction = jnp.where(is_descent, solution.value, -gradient / (1.0 + shift))

        y_scale = self.atol + self.rtol * jnp.abs(flat_eval)
        f_scale = self.atol + self.rtol * jnp.abs(f_eval)
        converged = (self.norm(trial_step / y_scale) < 1) & (
            jnp.abs(f_eval - old.f) < f_scale
        )
        new_state = _SparseState(
            first_step=jnp.array(False),
            y_eval=unravel(flat_new_y + direction),
            f_info=f_info,
            aux=aux,
            damping=damping,
            terminate=converged & jnp.invert(state.first_step),
            result=optx.RESULTS.successful,
        )
        input_aux = tree_where(state.first_step, aux, state.aux)
        return unravel(flat_new_y), new_state, input_aux

    def terminate(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        args: PyTree,
        options: dict[str, Any],
        state: _SparseState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        """Return whether the last step moved less than the tolerances allow."""
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Callable,
        y: PyTree[Float[Array, "..."]],
        aux: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: _SparseState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[PyTree[Float[Array, "..."]], PyTree, dict[str, Any]]:
        """Return the solution unchanged."""
        return y, aux, {}
