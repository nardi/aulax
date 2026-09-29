"""Constrained nonlinear optimization in JAX with the augmented Lagrangian method.

The conventional import style is `import aulax as alx`.
"""

import importlib.metadata

from ._problem import (
    Constraints as Constraints,
    Evaluation as Evaluation,
    ProblemFunction as ProblemFunction,
)


__version__ = importlib.metadata.version("aulax")
