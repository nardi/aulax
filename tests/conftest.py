import jax


# The tests compare against known solutions to tight tolerances, which float32
# cannot reach once the penalty grows large.
jax.config.update("jax_enable_x64", True)
