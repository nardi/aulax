import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


class Bounds(eqx.Module):
    r"""A box $l \le y \le u$ on the variables of a problem.

    Each field is `None` or a pytree whose leaves are scalars or arrays that
    broadcast to the leaves of `y`. A `None` field and an infinite entry both mean
    that there is no bound. The bounds are not part of the augmented Lagrangian,
    and the inner minimiser keeps every iterate inside the box.
    """

    lower: PyTree[Float[Array, "..."]] | None = None
    """The lower bound on `y`, or `None` for no lower bounds."""
    upper: PyTree[Float[Array, "..."]] | None = None
    """The upper bound on `y`, or `None` for no upper bounds."""

    def expand(self, y: PyTree[Float[Array, "..."]]) -> "Bounds":
        """Return the bounds as two pytrees that have exactly the shape of `y`.

        Missing bounds become negative or positive infinity, and scalar leaves are
        broadcast to the shape of the matching leaf of `y`. The leaves have the
        dtype of the leaves of `y`.
        """

        def expand_one(bound: PyTree | None, infinity: float) -> PyTree:
            if bound is None:
                return jax.tree.map(lambda leaf: jnp.full_like(leaf, infinity), y)
            return jax.tree.map(
                lambda leaf, value: jnp.broadcast_to(
                    jnp.asarray(value, leaf.dtype), leaf.shape
                ),
                y,
                bound,
            )

        return Bounds(
            lower=expand_one(self.lower, -jnp.inf),
            upper=expand_one(self.upper, jnp.inf),
        )

    def clip(self, y: PyTree[Float[Array, "..."]]) -> PyTree[Float[Array, "..."]]:
        """Return `y` projected onto the box, which is the nearest point inside it."""
        expanded = self.expand(y)
        return jax.tree.map(jnp.clip, y, expanded.lower, expanded.upper)
