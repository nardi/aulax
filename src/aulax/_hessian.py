import warnings
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree, Scalar


class Indices:
    """The row and column indices of a sparsity pattern, kept as host arrays.

    A pattern has to be known when the solver is traced, so it must not turn into a
    traced array when it is passed through `jit` as part of a solver. This class
    compares and hashes by content, which lets it sit in a static field.
    """

    def __init__(self, rows: Any, cols: Any):
        self.rows = np.asarray(rows, dtype=np.int32)
        """The row index of every entry."""
        self.cols = np.asarray(cols, dtype=np.int32)
        """The column index of every entry."""

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Indices)
            and np.array_equal(self.rows, other.rows)
            and np.array_equal(self.cols, other.cols)
        )

    def __hash__(self) -> int:
        return hash((self.rows.tobytes(), self.cols.tobytes()))


def _as_indices(pattern: Any) -> Indices:
    """Convert a sparsity pattern in any supported form to `Indices`.

    The supported forms are a pair of index arrays, a `BCOO` matrix, an object with
    `rows` and `cols` attributes like `asdex.SparsityPattern`, and a dense array
    whose nonzero entries mark the pattern.
    """
    if isinstance(pattern, Indices):
        return pattern
    if isinstance(pattern, tuple) and len(pattern) == 2:
        return Indices(*pattern)
    if isinstance(pattern, BCOO):
        indices = np.asarray(pattern.indices)
        return Indices(indices[:, 0], indices[:, 1])
    if hasattr(pattern, "rows") and hasattr(pattern, "cols"):
        return Indices(pattern.rows, pattern.cols)
    rows, cols = np.nonzero(np.asarray(pattern))
    return Indices(rows, cols)


class ProblemSparsity(eqx.Module):
    r"""What the user knows about the sparsity of their problem function.

    Both patterns are structural, which means that an entry is part of the pattern
    when the output depends on the input there, whatever its value at a given point.
    A product $y_1 y_2$ at $y = 0$ has a zero Jacobian and still depends on both.
    The columns follow `jax.flatten_util.ravel_pytree(y)`, and the rows of the
    constraint Jacobian follow `jax.tree.leaves((equality, inequality))`.

    The Hessian of the augmented Lagrangian is the Hessian of the objective, plus
    multiples of the constraint Hessians, plus the penalty times $J^\top J$. A
    constraint Hessian can only be nonzero where the constraint depends on both
    variables, which is inside the pattern of $J^\top J$. So these two patterns
    determine the pattern of the augmented Lagrangian.

    A pattern is a pair of index arrays, a `BCOO` matrix, an `asdex.SparsityPattern`
    or a dense array with nonzero entries at the pattern. If one of the two is
    missing, the sparse minimiser detects the whole pattern instead.
    """

    objective_hessian: Indices | None = eqx.field(default=None, static=True)
    """The pattern of the Hessian of the objective, or of the Lagrangian."""
    constraint_jacobian: Indices | None = eqx.field(default=None, static=True)
    """The pattern of the Jacobian of the stacked equality and inequality residuals."""

    def __init__(self, objective_hessian: Any = None, constraint_jacobian: Any = None):
        self.objective_hessian = (
            None if objective_hessian is None else _as_indices(objective_hessian)
        )
        self.constraint_jacobian = (
            None if constraint_jacobian is None else _as_indices(constraint_jacobian)
        )


_MAXIMUM_PRODUCT_ENTRIES = 50_000_000
"""The most entries that the pattern of $J^\\top J$ may produce before it is dense."""


def augmented_lagrangian_pattern(
    sparsity: ProblemSparsity, size: int
) -> Indices | None:
    r"""Return the Hessian pattern of the augmented Lagrangian, or `None` if unknown.

    This is the pattern of the objective Hessian united with the pattern of
    $J^\top J$, with both triangles. `None` is also the result when the product
    pattern would be so large that the problem is dense for practical purposes.
    """
    if sparsity.objective_hessian is None or sparsity.constraint_jacobian is None:
        return None
    hessian = sparsity.objective_hessian
    jacobian = sparsity.constraint_jacobian
    order = np.argsort(jacobian.rows, kind="stable")
    sorted_rows, sorted_cols = jacobian.rows[order], jacobian.cols[order]
    boundaries = np.flatnonzero(np.diff(sorted_rows)) + 1
    groups = np.split(sorted_cols, boundaries)
    if sum(len(group) ** 2 for group in groups) > _MAXIMUM_PRODUCT_ENTRIES:
        return None
    keys = [hessian.rows.astype(np.int64) * size + hessian.cols]
    keys.append(hessian.cols.astype(np.int64) * size + hessian.rows)
    for group in groups:
        group = np.unique(group).astype(np.int64)
        keys.append((group[:, None] * size + group[None, :]).ravel())
    unique = np.unique(np.concatenate(keys))
    return Indices(unique // size, unique % size)


# Steps of the trust region.


def trust_region_step(
    hessian: Float[Array, "n n"], gradient: Float[Array, " n"], radius: Scalar
) -> Float[Array, " n"]:
    r"""Return the minimizer of the quadratic model within the trust region.

    The model is $g^\top p + \frac{1}{2} p^\top H p$ with
    $\lVert p \rVert_2 \le \Delta$. The step is the Newton step when $H$ is
    positive definite and the step fits in the region. Otherwise it is
    $-(H + \sigma I)^{-1} g$ with the shift $\sigma$ that makes $H + \sigma I$
    positive definite and the step as long as the radius. An eigendecomposition
    of $H$ reduces the search for $\sigma$ to a one-dimensional bisection. The
    case where the gradient is orthogonal to the eigenvector of the smallest
    eigenvalue is not treated separately, and the step then stops short of the
    boundary.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    projected = eigenvectors.T @ gradient
    largest = jnp.max(jnp.abs(eigenvalues))
    # Eigenvalues below this are rounding noise, and count as zero curvature.
    floor = jnp.finfo(hessian.dtype).eps * largest + jnp.finfo(hessian.dtype).tiny

    def step_length(shift: Scalar) -> Scalar:
        return jnp.linalg.norm(projected / (eigenvalues + shift))

    positive_definite = eigenvalues[0] > floor
    newton_fits = positive_definite & (step_length(0.0) <= radius)
    lower = jnp.maximum(0.0, -eigenvalues[0]) + floor
    upper = lower + jnp.linalg.norm(gradient) / radius + largest

    def bisect(_, bounds: tuple[Scalar, Scalar]) -> tuple[Scalar, Scalar]:
        low, high = bounds
        middle = 0.5 * (low + high)
        too_long = step_length(middle) > radius
        return jnp.where(too_long, middle, low), jnp.where(too_long, high, middle)

    low, high = jax.lax.fori_loop(0, 100, bisect, (lower, upper))
    shift = jnp.where(newton_fits, 0.0, 0.5 * (low + high))
    return -eigenvectors @ (projected / (eigenvalues + shift))


ShiftedSolve = Callable[[Scalar], tuple[Array, Array, Array]]
"""Maps a shift $\\sigma$ to $p$, to $(H + \\sigma I)^{-1} p$, and to a flag.

The first output is $p = -(H + \\sigma I)^{-1} g$, the second is the solution of the
same system with $p$ on the right-hand side, and the flag tells whether
$H + \\sigma I$ is positive definite.
"""


def shifted_trust_region_step(
    solve: ShiftedSolve,
    gradient: Float[Array, " n"],
    radius: Scalar,
    matrix_scale: Scalar,
    max_iterations: int = 30,
) -> Float[Array, " n"]:
    r"""Return the trust-region step from solves with shifted matrices.

    This finds the shift $\sigma$ with $\lVert p(\sigma) \rVert = \Delta$ by Newton's
    method on $1 / \lVert p \rVert - 1 / \Delta$, safeguarded by bisection. Each
    iteration needs one factorization, and the second solve reuses it for the
    derivative. A shift that does not give a positive definite matrix counts as too
    small. If the iteration does not reach the boundary, the best step inside the
    region is returned, and with none a scaled steepest descent step.
    """
    dtype = gradient.dtype
    gradient_norm = jnp.linalg.norm(gradient)
    upper = gradient_norm / radius + jnp.maximum(matrix_scale, 1.0) * 10.0

    newton, _, newton_positive = solve(jnp.zeros((), dtype))
    newton_fits = newton_positive & (jnp.linalg.norm(newton) <= radius)

    def cond(carry):
        iteration, _, _, _, _, done = carry
        return (iteration < max_iterations) & jnp.invert(done)

    def body(carry):
        iteration, shift, low, high, best, _ = carry
        step, correction, positive = solve(shift)
        length = jnp.linalg.norm(step)
        curvature = step @ correction
        low = jnp.where(jnp.invert(positive) | (length > radius), shift, low)
        high = jnp.where(positive & (length <= radius), shift, high)
        within = positive & (jnp.abs(length - radius) <= 1e-2 * radius)
        newton_shift = shift - (1.0 / length - 1.0 / radius) / jnp.where(
            curvature > 0, curvature / length**3, jnp.inf
        )
        use_newton = (
            positive
            & jnp.isfinite(newton_shift)
            & (newton_shift > low)
            & (newton_shift < high)
        )
        next_shift = jnp.where(use_newton, newton_shift, 0.5 * (low + high))
        best = jnp.where(positive & (length <= 1.01 * radius), step, best)
        done = within | (high - low <= 1e-12 * upper)
        return iteration + 1, next_shift, low, high, best, done

    fallback = (
        -gradient
        * jnp.minimum(1.0, radius / jnp.maximum(gradient_norm, 1e-300))
        / jnp.maximum(1.0, matrix_scale)
    )
    start = jnp.minimum(gradient_norm / radius, upper)
    init = (
        jnp.array(0),
        start,
        jnp.zeros((), dtype),
        upper,
        fallback,
        jnp.array(False),
    )
    root = jax.lax.while_loop(cond, body, init)[4]
    return jnp.where(newton_fits, newton, root)


# Preconditioners for the matrix-free step.


class Preconditioner(Protocol):
    """Builds an approximate inverse of the Hessian at an accepted point."""

    def build(
        self, hessian_vector_product: Callable[[Array], Array], point: Array
    ) -> Callable[[Array], Array]:
        """Return a function that applies the approximate inverse to a vector.

        `hessian_vector_product` applies the Hessian at `point`. The approximate
        inverse must be symmetric and positive definite. It is built at every step
        of the minimiser, so it should be cheap compared with the conjugate gradient
        iterations that follow.
        """
        ...


class IdentityPreconditioner(eqx.Module):
    """No preconditioning."""

    def build(
        self, hessian_vector_product: Callable[[Array], Array], point: Array
    ) -> Callable[[Array], Array]:
        """Return the identity."""
        return lambda vector: vector


class JacobiPreconditioner(eqx.Module):
    """The inverse of the diagonal of the Hessian.

    The diagonal is computed with one Hessian-vector product per variable, or
    estimated with `probes` random sign vectors, whose products have the expected
    value of the diagonal. Entries are replaced by their absolute value and kept
    away from zero, so that the preconditioner is positive definite.
    """

    probes: int | None = eqx.field(default=None, static=True)
    """The number of random probes, or `None` for the exact diagonal."""
    floor: float = 1e-8
    """The smallest entry, relative to the largest, that the diagonal may have."""

    def build(
        self, hessian_vector_product: Callable[[Array], Array], point: Array
    ) -> Callable[[Array], Array]:
        """Return division by the diagonal."""
        size = point.size
        if self.probes is None:

            def entry(index: Array) -> Array:
                unit = jnp.zeros(size, point.dtype).at[index].set(1.0)
                return hessian_vector_product(unit)[index]

            diagonal = jax.vmap(entry)(jnp.arange(size))
        else:
            signs = jax.random.rademacher(
                jax.random.key(0), (self.probes, size), dtype=point.dtype
            )
            products = jax.vmap(hessian_vector_product)(signs)
            diagonal = jnp.mean(signs * products, axis=0)
        magnitude = jnp.abs(diagonal)
        magnitude = jnp.maximum(magnitude, self.floor * jnp.max(magnitude) + 1e-300)
        return lambda vector: vector / magnitude


def _steihaug(
    hessian_vector_product: Callable[[Array], Array],
    apply_inverse: Callable[[Array], Array],
    gradient: Array,
    radius: Scalar,
    max_iterations: int,
) -> Array:
    """Steihaug-Toint conjugate gradients inside the trust region."""
    gradient_norm = jnp.linalg.norm(gradient)
    tolerance = gradient_norm * jnp.minimum(0.1, gradient_norm)

    def boundary(point: Array, direction: Array) -> Array:
        a = direction @ direction
        b = 2.0 * point @ direction
        c = point @ point - radius**2
        root = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 0.0))
        return point + (-b + root) / (2.0 * jnp.maximum(a, 1e-300)) * direction

    def cond(carry):
        iteration, _, _, _, _, done = carry
        return (iteration < max_iterations) & jnp.invert(done)

    def body(carry):
        iteration, z, r, d, preconditioned, _ = carry
        product = hessian_vector_product(d)
        curvature = d @ product
        negative = curvature <= 0
        alpha = (r @ preconditioned) / jnp.where(negative, 1.0, curvature)
        z_next = z + alpha * d
        outside = jnp.linalg.norm(z_next) >= radius
        r_next = r + alpha * product
        preconditioned_next = apply_inverse(r_next)
        beta = (r_next @ preconditioned_next) / jnp.maximum(r @ preconditioned, 1e-300)
        d_next = -preconditioned_next + beta * d
        stop = negative | outside
        z_out = jnp.where(stop, boundary(z, d), z_next)
        converged = jnp.linalg.norm(r_next) <= tolerance
        return (
            iteration + 1,
            z_out,
            r_next,
            d_next,
            preconditioned_next,
            stop | converged,
        )

    start = apply_inverse(gradient)
    init = (
        jnp.array(0),
        jnp.zeros_like(gradient),
        gradient,
        -start,
        start,
        gradient_norm == 0,
    )
    return jax.lax.while_loop(cond, body, init)[1]


# Prepared Hessians. A strategy returns one when the minimiser is traced.


class PreparedHessian(Protocol):
    """What the minimiser needs from a strategy, once the problem is known."""

    mode: str
    """The name of the mode, `"dense"`, `"sparse"` or `"matrix_free"`."""
    data_shape: tuple[int, ...]
    """The shape of the array that represents the Hessian at an accepted point."""
    fused: bool
    """Whether `evaluate_fused` computes the value, gradient and Hessian in one pass.

    That costs one evaluation of the problem per trial point, but it computes the
    Hessian at rejected points too. Without it, the Hessian is computed in a second
    pass at accepted points only.
    """

    def evaluate(self, scalar: Callable[[Array], Scalar], flat_y: Array) -> Array:
        """Return the Hessian data of `scalar` at `flat_y`.

        `scalar` is the objective of the current inner problem as a function of the
        flat point. It changes with the multipliers and the penalty, so a prepared
        Hessian holds structure only and receives it at every call.
        """
        ...

    def matvec(
        self, scalar: Callable[[Array], Scalar], data: Array, vector: Array
    ) -> Array:
        """Apply the Hessian described by `data` to `vector`."""
        ...

    def step(
        self,
        scalar: Callable[[Array], Scalar],
        data: Array,
        gradient: Array,
        radius: Scalar,
    ) -> Array:
        """Return the step inside the trust region."""
        ...


def _scalar_function(fn: Callable, unravel: Callable, args: PyTree) -> Callable:
    """Return the objective of the inner problem as a function of the flat point."""
    return lambda flat, arguments=args: fn(unravel(flat), arguments)[0]


def _without_debug_callbacks(function: Callable, sample: Array) -> Callable:
    """Return `function` with its top-level `jax.debug.callback` calls removed.

    `asdex` cannot trace a debug callback, but the callback has no outputs, so
    dropping it from the jaxpr leaves the values unchanged. This lets a problem
    that counts or prints its evaluations still get a sparsity pattern.
    """
    closed_jaxpr = jax.make_jaxpr(function)(sample)
    kept_equations = [
        equation
        for equation in closed_jaxpr.jaxpr.eqns
        if equation.primitive.name != "debug_callback"
    ]
    jaxpr = closed_jaxpr.jaxpr.replace(eqns=kept_equations)
    return lambda flat: jax.core.eval_jaxpr(jaxpr, closed_jaxpr.consts, flat)[0]


class _DensePrepared:
    mode = "dense"

    def __init__(self, size, dtype, eigendecomposition_limit):
        self.size = size
        self.dtype = dtype
        self.use_eigendecomposition = size <= eigendecomposition_limit
        self.data_shape = (size, size)

    fused = True

    def evaluate(self, scalar, flat_y):
        hessian = jax.hessian(scalar)(flat_y)
        return 0.5 * (hessian + hessian.T)

    def evaluate_fused(self, function, flat_y):
        def gradient_with_values(flat):
            (value, aux), gradient = jax.value_and_grad(function, has_aux=True)(flat)
            return gradient, (value, aux, gradient)

        hessian, (value, aux, gradient) = jax.jacfwd(
            gradient_with_values, has_aux=True
        )(flat_y)
        return value, aux, gradient, 0.5 * (hessian + hessian.T)

    def matvec(self, scalar, data, vector):
        return data @ vector

    def step(self, scalar, data, gradient, radius):
        if self.use_eigendecomposition:
            return trust_region_step(data, gradient, radius)
        identity = jnp.eye(self.size, dtype=data.dtype)

        def solve(shift):
            factor = jnp.linalg.cholesky(data + shift * identity)
            positive = jnp.all(jnp.isfinite(factor))
            step = -jax.scipy.linalg.cho_solve((factor, True), gradient)
            correction = jax.scipy.linalg.cho_solve((factor, True), step)
            return step, correction, positive

        scale = jnp.max(jnp.abs(jnp.diag(data)))
        return shifted_trust_region_step(solve, gradient, radius, scale)


class _MatrixFreePrepared:
    mode = "matrix_free"
    fused = False

    def __init__(self, size, dtype, preconditioner, max_iterations):
        self.size = size
        self.preconditioner = preconditioner
        self.max_iterations = (
            min(2 * size + 20, 500) if max_iterations is None else max_iterations
        )
        self.data_shape = (size,)

    def evaluate(self, scalar, flat_y):
        return flat_y

    def matvec(self, scalar, data, vector):
        return jax.jvp(jax.grad(scalar), (data,), (vector,))[1]

    def step(self, scalar, data, gradient, radius):
        def product(vector):
            return self.matvec(scalar, data, vector)

        apply_inverse = self.preconditioner.build(product, data)
        return _steihaug(product, apply_inverse, gradient, radius, self.max_iterations)


class _SparsePrepared:
    mode = "sparse"
    fused = False

    def __init__(self, size, dtype, solver, coloring, check):
        import splineax

        self.splineax = splineax
        self.size = size
        self.solver = solver
        self.coloring = coloring
        self.check = check
        pattern = coloring.sparsity
        self.rows = np.asarray(pattern.rows, dtype=np.int64)
        self.cols = np.asarray(pattern.cols, dtype=np.int64)
        diagonal = np.arange(size, dtype=np.int64)
        keys = np.concatenate(
            [self.rows * size + self.cols, diagonal * size + diagonal]
        )
        union, inverse = np.unique(keys, return_inverse=True)
        self.union = np.stack([union // size, union % size], axis=1).astype(np.int32)
        self.hessian_positions = inverse[: self.rows.size]
        self.diagonal_positions = inverse[self.rows.size :]
        self.entry_indices = np.stack([self.rows, self.cols], axis=1).astype(np.int32)
        self.data_shape = (self.rows.size,)
        tag_pattern = BCOO((np.ones(len(self.union)), self.union), shape=(size, size))
        self.tag = splineax.sparsity_pattern_tag(tag_pattern)
        self.dtype = dtype

    def evaluate(self, scalar, flat_y):
        import asdex

        function = asdex.hessian_from_coloring(scalar, self.coloring)
        data = function(flat_y).data
        if self.check:
            dense = jax.hessian(scalar)(flat_y)
            sparse = BCOO((data, self.entry_indices), shape=(self.size,) * 2)
            difference = jnp.max(jnp.abs(sparse.todense() - dense))
            data = eqx.error_if(
                data,
                difference > 1e-8 * (1.0 + jnp.max(jnp.abs(dense))),
                "The sparsity pattern misses entries of the Hessian of the "
                "augmented Lagrangian.",
            )
        return data

    def matvec(self, scalar, data, vector):
        matrix = BCOO((data, self.entry_indices), shape=(self.size,) * 2)
        return matrix @ vector

    def _operator(self, data, shift):
        # A matrix with an empty row is exactly singular, which KLU reports as an
        # error. A shift at rounding level avoids that.
        floor = 1e-12 * jnp.maximum(jnp.max(jnp.abs(data), initial=0.0), 1.0)
        values = (
            jnp.zeros(len(self.union), data.dtype)
            .at[self.hessian_positions]
            .add(data)
            .at[self.diagonal_positions]
            .add(jnp.maximum(shift, floor))
        )
        matrix = BCOO(
            (values, self.union),
            shape=(self.size,) * 2,
            indices_sorted=True,
            unique_indices=True,
        )
        return self.splineax.BCOOLinearOperator(matrix, tags=self.tag)

    def _solve(self, operator, rhs):
        return lx.linear_solve(operator, rhs, self.solver, throw=False).value

    def step(self, scalar, data, gradient, radius):
        def shifted(data, gradient, radius):
            def solve(shift):
                operator = self._operator(data, shift)
                step = -self._solve(operator, gradient)
                correction = self._solve(operator, step)
                # Without the inertia of the factorization, a descent direction with
                # positive curvature stands in for a positive definite matrix.
                positive = (
                    jnp.all(jnp.isfinite(step))
                    & (gradient @ step < 0)
                    & (step @ correction > 0)
                )
                return step, correction, positive

            diagonal = (
                jnp.zeros(self.size, data.dtype)
                .at[self.rows]
                .add(jnp.where(self.rows == self.cols, jnp.abs(data), 0.0))
            )
            return shifted_trust_region_step(solve, gradient, radius, jnp.max(diagonal))

        return self.splineax.stateful_solve_transform(shifted)(data, gradient, radius)


# Strategies.


class HessianStrategy(Protocol):
    """Decides how the Hessian of the augmented Lagrangian is represented and used."""

    def prepare(self, fn: Callable, y: PyTree, args: PyTree) -> PreparedHessian:
        """Return the prepared Hessian for the inner problem `fn` at `y`."""
        ...


def _sample_args(args: PyTree) -> PyTree:
    """Return `args` with ones in place of every array.

    Sparsity detection folds constants, so zeros would turn divisions by the
    penalty into divisions by zero.
    """
    return jax.tree.map(
        lambda leaf: (
            np.ones(jnp.shape(leaf), jnp.result_type(leaf))
            if isinstance(leaf, jax.Array)
            else leaf
        ),
        args,
    )


class DenseHessian(eqx.Module):
    """Forms the full Hessian and takes trust-region steps with dense linear algebra.

    Up to `eigendecomposition_limit` variables the step comes from an
    eigendecomposition of the Hessian, which is exact for indefinite matrices and
    gives every shift at once. Beyond that it comes from Cholesky factorizations
    with a search for the shift, where a failed factorization tells that the matrix
    is not positive definite.
    """

    eigendecomposition_limit: int = eqx.field(default=150, static=True)
    """The largest number of variables that uses an eigendecomposition."""

    def prepare(self, fn: Callable, y: PyTree, args: PyTree) -> PreparedHessian:
        """Return the dense Hessian."""
        flat, _ = ravel_pytree(y)
        return _DensePrepared(flat.size, flat.dtype, self.eigendecomposition_limit)


class MatrixFreeHessian(eqx.Module):
    """Applies the Hessian through Hessian-vector products and never forms it.

    The step comes from preconditioned conjugate gradients inside the trust region,
    which stop at the boundary or at a direction of negative curvature.
    """

    preconditioner: Any = None
    """A `Preconditioner`, or a function `(hvp, point) -> apply` that builds one.

    `None` means no preconditioning.
    """
    max_iterations: int | None = eqx.field(default=None, static=True)
    """The most conjugate gradient iterations per step, or `None` for a default."""

    def prepare(self, fn: Callable, y: PyTree, args: PyTree) -> PreparedHessian:
        """Return the matrix-free Hessian."""
        flat, _ = ravel_pytree(y)
        preconditioner = self.preconditioner
        if preconditioner is None:
            preconditioner = IdentityPreconditioner()
        elif not hasattr(preconditioner, "build"):
            function = preconditioner

            class _FromFunction:
                def build(self, hessian_vector_product, point):
                    return function(hessian_vector_product, point)

            preconditioner = _FromFunction()
        return _MatrixFreePrepared(
            flat.size, flat.dtype, preconditioner, self.max_iterations
        )


class SparseHessian(eqx.Module):
    """Forms the Hessian as a sparse matrix and takes steps with a sparse solver.

    The sparsity pattern comes from `sparsity`, or from detection with `asdex` when
    it is not given. The Hessian is computed with as many Hessian-vector products as
    the coloring of the pattern needs. It needs the optional packages `asdex` and
    `splineax` (`pip install aulax[sparse]`).
    """

    sparsity: ProblemSparsity | None = None
    """What the user knows about the pattern, or `None` to detect it."""
    linear_solver: Any = eqx.field(default=None, static=True)
    """The `splineax` solver, or `None` for `splineax.AutoSparseLinearSolver`."""
    check_pattern: bool = eqx.field(default=False, static=True)
    """Whether to compare the sparse Hessian with a dense one and raise on a mismatch.

    This is a debugging aid, since it computes a dense Hessian at every step.
    """

    def prepare(self, fn: Callable, y: PyTree, args: PyTree) -> PreparedHessian:
        """Return the sparse Hessian, detecting its pattern if needed."""
        asdex, splineax = _import_sparse_libraries()
        flat, unravel = ravel_pytree(y)
        size = flat.size
        indices = (
            None
            if self.sparsity is None
            else augmented_lagrangian_pattern(self.sparsity, size)
        )
        if indices is not None:
            pattern = asdex.SparsityPattern.from_coo(
                indices.rows, indices.cols, (size, size)
            )
            coloring = asdex.hessian_coloring_from_sparsity(pattern)
        else:
            sample = _sample_args(args)
            scalar = _without_debug_callbacks(
                _scalar_function(fn, unravel, sample), np.zeros(size, flat.dtype)
            )
            coloring = asdex.hessian_coloring(scalar, np.zeros(size, flat.dtype))
        solver = (
            splineax.AutoSparseLinearSolver()
            if self.linear_solver is None
            else self.linear_solver
        )
        return _SparsePrepared(size, flat.dtype, solver, coloring, self.check_pattern)


def _import_sparse_libraries():
    """Import `asdex` and `splineax`, which sparse mode needs."""
    try:
        import asdex
        import splineax
    except ImportError as error:
        raise ImportError(
            "Sparse mode needs the `asdex` and `splineax` packages. Install them "
            "with `pip install aulax[sparse]`."
        ) from error
    return asdex, splineax


class AutoHessian(eqx.Module):
    """Chooses between the dense, sparse and matrix-free strategies.

    Problems with at most `dense_limit` variables use `DenseHessian`. Larger ones try
    `SparseHessian`, and accept it when the detection works and the pattern is
    sparse enough for a factorization to pay off. Everything else uses
    `MatrixFreeHessian`. The choice is made when the solver is traced, and a warning
    names the reason when sparse mode is rejected.
    """

    dense_limit: int = eqx.field(default=150, static=True)
    """The largest number of variables that uses dense mode."""
    sparsity: ProblemSparsity | None = None
    """What the user knows about the sparsity pattern, passed on to sparse mode."""
    linear_solver: Any = eqx.field(default=None, static=True)
    """The sparse solver, passed on to sparse mode."""
    preconditioner: Any = None
    """The preconditioner, passed on to matrix-free mode."""
    max_density: float = eqx.field(default=0.05, static=True)
    """The largest fraction of nonzero entries for which sparse mode is accepted."""
    max_color_fraction: float = eqx.field(default=0.1, static=True)
    """The largest number of colors, as a fraction of the variables, for sparse mode."""

    def prepare(self, fn: Callable, y: PyTree, args: PyTree) -> PreparedHessian:
        """Return the prepared Hessian of the first mode that suits the problem."""
        flat, _ = ravel_pytree(y)
        size = flat.size
        if size <= self.dense_limit:
            return DenseHessian().prepare(fn, y, args)
        reason = None
        try:
            sparse = SparseHessian(
                sparsity=self.sparsity, linear_solver=self.linear_solver
            ).prepare(fn, y, args)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
        else:
            entries = sparse.rows.size
            colors = sparse.coloring.num_colors
            if entries > self.max_density * size**2:
                reason = f"the Hessian pattern is {entries / size**2:.0%} dense"
            elif colors > self.max_color_fraction * size:
                reason = f"its coloring needs {colors} colors for {size} variables"
            else:
                return sparse
        warnings.warn(
            f"Sparse mode was rejected ({reason}), so the minimiser is matrix-free.",
            stacklevel=2,
        )
        return MatrixFreeHessian(preconditioner=self.preconditioner).prepare(
            fn, y, args
        )
