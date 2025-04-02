import math
import types

import numpy as np
from numpy import ndarray

from sparkle.src.agent.base import BaseAgent
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.mlhs import MLHS
from sparkle.src.utils.default import set_default


###############################################
class CEM(BaseAgent):
    """
    Cross-Entropy Method (CEM) agent.

    This agent implements the Cross-Entropy Method, a stochastic optimization
    technique that iteratively refines a sampling distribution to find the
    optimum of a function.
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: types.SimpleNamespace) -> None:
        """
        Initializes the CEM agent.

        Args:
            path: The base path for storing results.
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the agent.
        """
        super().__init__(path, spaces, pms)

        self.name        = "CEM"
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_points    = set_default("n_points", 2*self.dim, pms)
        self.n_elites    = set_default("n_elites", math.floor(self.n_points/2), pms)
        self.alpha       = set_default("alpha", 0.2, pms)

        if (not self.silent): self.summary()

    def reset(self, run: int) -> None:
        """
        Resets the CEM agent for a new run.

        Args:
            run: The run number.
        """

        # Mother class reset
        super().reset(run)

        # Min and max arrays used for cem adaptation
        self.xmin_cem = self.xmin.copy()
        self.xmax_cem = self.xmax.copy()

    def sample(self) -> ndarray:
        """
        Samples new points from the CEM distribution.

        This method generates new points based on the current elite set,
        adapting the sampling distribution towards the region of better
        performing points.

        Returns:
            A NumPy array of shape (n_points, dim) representing the new points.
        """
        pms          = types.SimpleNamespace()
        pms.n_points = self.n_points
        pms.n_iter   = 1000

        spaces = {"dim": self.spaces.dim,
                  "xmin": self.xmin_cem,
                  "xmax": self.xmax_cem}

        spaces = EnvSpaces(spaces)
        pex    = MLHS(spaces, pms)

        return pex.x

    def step(self, x: ndarray, c: ndarray) -> None:
        """
        Performs one step of the CEM algorithm.

        This method updates the sampling distribution based on the elite set
        and increments the step counter.

        Args:
            x: The points that were evaluated.
            c: The cost values at the evaluated points.
        """

        # Sort
        self.sort(x, c)

        # Update xmin and xmax
        xmin = np.amin(x[:self.n_elites,:], axis=0)
        xmax = np.amax(x[:self.n_elites,:], axis=0)
        self.xmin_cem[:] = ((1.0-self.alpha)*self.xmin_cem[:] + self.alpha*xmin[:])
        self.xmax_cem[:] = ((1.0-self.alpha)*self.xmax_cem[:] + self.alpha*xmax[:])

        self.stp += 1

    def sort(self, x: ndarray, c: ndarray) -> None:
        """
        Sorts the offsprings based on their cost values.

        Args:
            x: The points that were evaluated.
            c: The cost values at the evaluated points.
        """

        sc   = np.argsort(c)
        x[:] = x[sc[:]]
        c[:] = c[sc[:]]
