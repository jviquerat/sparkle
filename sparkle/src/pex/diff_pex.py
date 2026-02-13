from types import SimpleNamespace
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import jit, value_and_grad

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.diff_costs import cost_factory
from sparkle.src.pex.mlhs import MLHS
from sparkle.src.utils.default import set_default
from sparkle.src.utils.prints import fmt_float, spacer


###############################################
class DiffPex(BasePex):
    """
    Differentiable design of experiment implementation.

    This class implements a design of experiments where the points are optimized
    using gradient descent (L-BFGS-B via JAX/SciPy) with a swappable cost function.
    It starts from an initial distribution (default: maximin LHS) and optimizes
    the position of sample points.
    """

    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the differentiable experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters.
                 - cost: Name of the cost function (default: "maximin").
                 - cost_fn: Optional callable for a custom cost function.
        """
        super().__init__(spaces, pms)

        self.name = "differentiable"
        self.cost_name = set_default("cost", "maximin", pms)
        self.pms = pms

        # Determine cost object
        self.cost_fn = getattr(pms, "cost_fn", None)
        if self.cost_fn is None:
            self.cost_obj = cost_factory.create(self.cost_name,
                                                spaces=spaces,
                                                pms=pms)
            self.cost_fn = self.cost_obj

        # Precompute bounds for SciPy optimizer
        # Bounds are (N*D,) arrays for lower and upper limits
        self.lb = np.tile(self.xmin, self.n_points_)
        self.ub = np.tile(self.xmax, self.n_points_)
        self.bounds = scipy.optimize.Bounds(self.lb, self.ub)

        self.reset()

    def reset(self) -> None:
        """
        Resets the plan by initializing with MLHS and optimizing using JAX/SciPy.
        """
        # Start from maximin LHS to get a good initial distribution
        initial_design = MLHS(self.spaces, self.pms)
        x_init = initial_design.x

        # Define the objective function wrapper for SciPy (flat input)
        def loss(x_flat):
            x = x_flat.reshape((self.n_points_, self.dim))
            return self.cost_fn(x)

        # Create JIT-compiled value_and_grad function
        # We perform compilation here to ensure shapes are correct
        jit_val_and_grad = jit(value_and_grad(loss))

        # Optimization using SciPy L-BFGS-B
        # jac=True means the function returns (value, gradient)
        res = scipy.optimize.minimize(fun=jit_val_and_grad,
                                      x0=x_init.flatten(),
                                      method='L-BFGS-B',
                                      jac=True,
                                      bounds=self.bounds)

        # Store the result
        self.x_ = res.x.reshape((self.n_points_, self.dim))

    def summary(self) -> None:
        """
        Prints a summary of the experiment plan's configuration.
        """
        super().summary()
        spacer("Cost function: " + str(self.cost_name))
        spacer("Optimizer: SciPy L-BFGS-B (JAX backend)")