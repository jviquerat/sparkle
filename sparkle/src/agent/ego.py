from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
from numpy import ndarray

from sparkle.src.agent.base import BaseAgent
from sparkle.src.agent.ms_lbfgsb import MSLBFGSB
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.infill.infill import infill_factory
from sparkle.src.utils.default import set_default

###############################################
class EGO(BaseAgent):
    """
    Efficient Global Optimization (EGO) agent.

    This agent implements the EGO algorithm, which combines a surrogate model
    with an infill criterion to efficiently explore the search space and find
    the global optimum of a function.
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 model: Any,
                 pms: SimpleNamespace) -> None:
        """
        Initializes the EGO agent.

        Args:
            path: The base path for storing results.
            spaces: The environment's search space definition.
            model: The surrogate model to use for function approximation.
            pms: A SimpleNamespace object containing parameters for the agent.
        """
        super().__init__(path, spaces, pms)

        self.name        = "EGO"
        self.spaces      = spaces
        self.n_points    = 1
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.model       = model

        # Initialize infill
        self.infill = infill_factory.create(pms.infill,
                                            spaces = spaces,
                                            model  = model)

        self.summary()

    def reset(self, run: int) -> None:
        """
        Resets the EGO agent for a new run.

        Args:
            run: The run number.
        """

        super().reset(run)

    def best_point(self) -> Tuple[ndarray, float]:
        """
        Returns the best point found so far by the surrogate model.

        Returns:
            A tuple containing:
                - The best point (NumPy array).
                - The best function value at that point (float).
        """

        k = np.argmin(self.model.y)
        return self.model.x[k], self.model.y[k]

    def sample(self) -> ndarray:
        """
        Samples a new point based on the expected improvement criterion.

        This method optimizes the infill criterion to find a promising new
        point to evaluate.

        Returns:
            A NumPy array representing the new point to evaluate.
        """

        # Set best point to infill before optimization
        xb, yb  = self.best_point()
        self.infill.set_best(xb, yb)

        # Optimize
        f_lambda = lambda x: -self.infill(x)
        grad_lambda = lambda x: -self.infill.grad(x)[0]
        opt  = MSLBFGSB()
        x, c = opt.optimize(f_lambda,
                            self.spaces.xmin,
                            self.spaces.xmax,
                            df=grad_lambda,
                            n_pts=10*self.spaces.dim,
                            m=20,
                            tol=1.0e-6,
                            max_iter=200)

        return np.reshape(x, (-1,self.spaces.dim))

    def step(self, x: ndarray, c: ndarray) -> None:
        """
        Performs one step of the EGO algorithm.

        This method updates the step counter.

        Args:
            x: The point that was evaluated.
            c: The cost value at the evaluated point.
        """

        self.stp += 1

    def done(self) -> bool:
        """
        Checks if the EGO algorithm has reached its termination condition.

        Returns:
            True if the maximum number of steps has been reached, False otherwise.
        """

        if (self.stp == self.n_steps_max): return True
        return False
