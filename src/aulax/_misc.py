import functools
import operator
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar


def _weak_zero() -> Scalar:
    """Return a weakly typed zero, which takes the dtype of whatever it meets."""
    return jnp.asarray(0.0)


def tree_sum(function: Callable[..., Array], *trees: PyTree[Array]) -> Scalar:
    """Apply `function` leafwise to `trees` and return the sum of all results.

    Returns zero when the trees have no leaves, which happens for a constraint
    kind the problem does not have.
    """
    results = jax.tree.leaves(jax.tree.map(function, *trees))
    leaf_sums = [jnp.sum(result) for result in results]
    return functools.reduce(operator.add, leaf_sums, _weak_zero())


def tree_max_norm(tree: PyTree[Array]) -> Scalar:
    """Return the largest absolute value in `tree`, or zero if it has no leaves."""
    leaf_maxima = [
        jnp.max(jnp.abs(leaf), initial=0.0) for leaf in jax.tree.leaves(tree)
    ]
    return functools.reduce(jnp.maximum, leaf_maxima, _weak_zero())


def tree_any(tree: PyTree[Bool[Array, "..."]]) -> Bool[Array, ""]:
    """Return whether any element of `tree` is true, or false if it has no leaves."""
    leaf_results = [jnp.any(leaf) for leaf in jax.tree.leaves(tree)]
    return functools.reduce(jnp.logical_or, leaf_results, jnp.array(False))


def tree_where(
    condition: Bool[ArrayLike, ""], true_tree: PyTree, false_tree: PyTree
) -> PyTree:
    """Select `true_tree` where `condition` holds and `false_tree` elsewhere."""
    return jax.tree.map(
        lambda true, false: jnp.where(condition, true, false), true_tree, false_tree
    )


def tree_full_like(tree: PyTree[Array], value: ArrayLike) -> PyTree[Array]:
    """Return a tree shaped like `tree` with every element set to `value`."""
    return jax.tree.map(lambda leaf: jnp.full_like(leaf, value), tree)
