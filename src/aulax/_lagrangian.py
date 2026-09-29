import jax
import jax.numpy as jnp
from jaxtyping import Scalar

from ._misc import tree_max_norm, tree_sum
from ._problem import Constraints, Evaluation


def augmented_lagrangian(
    evaluation: Evaluation, multipliers: Constraints, penalty: Constraints
) -> Scalar:
    r"""Return the Powell-Hestenes-Rockafellar augmented Lagrangian.

    With one penalty $\rho_i$ per constraint element, this is

    $$
    f + \sum_i \left( \lambda_i h_i + \frac{\rho_i}{2} h_i^2 \right)
      + \sum_j \frac{\max(0, \mu_j + \rho_j g_j)^2 - \mu_j^2}{2 \rho_j}.
    $$
    """
    equality_terms = tree_sum(
        lambda residual, multiplier, rho: (
            multiplier * residual + 0.5 * rho * residual**2
        ),
        evaluation.equality,
        multipliers.equality,
        penalty.equality,
    )
    inequality_terms = tree_sum(
        lambda residual, multiplier, rho: (
            (jnp.maximum(0.0, multiplier + rho * residual) ** 2 - multiplier**2)
            / (2.0 * rho)
        ),
        evaluation.inequality,
        multipliers.inequality,
        penalty.inequality,
    )
    return evaluation.objective + equality_terms + inequality_terms


def lagrangian(evaluation: Evaluation, multipliers: Constraints) -> Scalar:
    r"""Return the Lagrangian $f + \lambda^\top h + \mu^\top g$."""
    equality_terms = tree_sum(jnp.multiply, evaluation.equality, multipliers.equality)
    inequality_terms = tree_sum(
        jnp.multiply, evaluation.inequality, multipliers.inequality
    )
    return evaluation.objective + equality_terms + inequality_terms


def penalty_infeasibility(
    evaluation: Evaluation, multipliers: Constraints, penalty: Constraints
) -> Constraints:
    r"""Return the elementwise infeasibility that the penalty update compares.

    For equality constraints this is $\lvert h_i \rvert$. For inequality
    constraints it is $\lvert \min(-g_j, \mu_j / \rho_j) \rvert$, which measures
    both violation and complementarity. It is small for an inactive constraint
    whose multiplier is near zero, and for an active constraint that holds.
    """
    return Constraints(
        equality=jax.tree.map(jnp.abs, evaluation.equality),
        inequality=jax.tree.map(
            lambda residual, multiplier, rho: jnp.abs(
                jnp.minimum(-residual, multiplier / rho)
            ),
            evaluation.inequality,
            multipliers.inequality,
            penalty.inequality,
        ),
    )


def kkt_infeasibility(evaluation: Evaluation, multipliers: Constraints) -> Scalar:
    r"""Return the largest violation of feasibility or complementarity.

    This is the larger of $\lVert h \rVert_\infty$ and
    $\lVert \min(-g, \mu) \rVert_\infty$. The second term covers both a violated
    inequality, where $g_j > 0$, and a positive multiplier on an inactive one.
    """
    complementarity = jax.tree.map(
        lambda residual, multiplier: jnp.minimum(-residual, multiplier),
        evaluation.inequality,
        multipliers.inequality,
    )
    return jnp.maximum(
        tree_max_norm(evaluation.equality), tree_max_norm(complementarity)
    )
