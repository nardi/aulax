"""Constrained nonlinear optimization in JAX with the augmented Lagrangian method.

The conventional import style is `import aulax as alx`.
"""

import importlib.metadata

from ._inner import (
    GradientStopping as GradientStopping,
    InnerStateAccess as InnerStateAccess,
    InnerStoppingRule as InnerStoppingRule,
    MinimiserStopping as MinimiserStopping,
    OptimistixStateAccess as OptimistixStateAccess,
)
from ._minimise import minimise as minimise
from ._newton import NewtonTrustRegion as NewtonTrustRegion
from ._problem import (
    Constraints as Constraints,
    Evaluation as Evaluation,
    ProblemFunction as ProblemFunction,
)
from ._scaling import (
    GradientScaling as GradientScaling,
    ProblemScaling as ProblemScaling,
    Scaling as Scaling,
)
from ._solution import (
    RESULTS as RESULTS,
    Solution as Solution,
    Statistics as Statistics,
)
from ._solver import AugmentedLagrangian as AugmentedLagrangian
from ._sparse_newton import SparseNewton as SparseNewton
from ._updates import (
    AlgencanPenaltyUpdate as AlgencanPenaltyUpdate,
    GeometricToleranceSchedule as GeometricToleranceSchedule,
    MultiplierUpdate as MultiplierUpdate,
    PenaltyUpdate as PenaltyUpdate,
    SafeguardedMultiplierUpdate as SafeguardedMultiplierUpdate,
    ToleranceSchedule as ToleranceSchedule,
)


__version__ = importlib.metadata.version("aulax")
