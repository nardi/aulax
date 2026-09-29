import jax
import jax.numpy as jnp

import aulax as alx


def test_evaluation_defaults_to_no_constraints_state_or_aux():
    evaluation = alx.Evaluation(objective=jnp.array(1.0))
    assert evaluation.equality is None
    assert evaluation.inequality is None
    assert evaluation.state is None
    assert evaluation.aux is None


def test_evaluation_is_a_pytree_of_its_arrays():
    evaluation = alx.Evaluation(
        objective=jnp.array(1.0),
        equality={"a": jnp.zeros(2), "b": jnp.zeros(())},
        inequality=jnp.ones(3),
        state=jnp.arange(4),
    )
    assert len(jax.tree.leaves(evaluation)) == 5


def test_plain_function_satisfies_problem_protocol():
    def problem(y: jax.Array, args: None, state: None) -> alx.Evaluation[None, None]:
        return alx.Evaluation(objective=jnp.sum(y**2), equality=y[0] - 1.0)

    protocol_problem: alx.ProblemFunction[jax.Array, None, None, None] = problem
    evaluation = protocol_problem(jnp.array([1.0, 2.0]), None, None)
    assert evaluation.objective == 5.0
