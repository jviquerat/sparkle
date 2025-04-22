from types import SimpleNamespace

import numpy as np
from numpy import ndarray

from sparkle.src.agent.base import BaseAgent
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default


###############################################
class PSO(BaseAgent):
    """
    Particle Swarm Optimization (PSO) agent.

    This agent implements the Particle Swarm Optimization algorithm, a
    population-based stochastic optimization technique inspired by the social
    behavior of bird flocking or fish schooling.
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: SimpleNamespace) -> None:
        """
        Initializes the PSO agent.

        Args:
            path: The base path for storing results.
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the agent.
        """
        super().__init__(path, spaces, pms)

        self.name        = "PSO"
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_points    = set_default("n_points", 20, pms)
        self.v0          = set_default("v0", 0.1, pms)
        self.c1          = set_default("c1", 0.5, pms)
        self.c2          = set_default("c2", 0.5, pms)
        self.w           = set_default("w", 0.8, pms)

        if (not self.silent): self.summary()

    def reset(self, run: int) -> None:
        """
        Resets the PSO agent for a new run.

        Args:
            run: The run number.
        """

        # Mother class reset
        super().reset(run)

        # Local best point
        # We store the best position of each particle
        self.p_best  = np.zeros((self.n_points, self.dim))
        self.p_score = np.ones(self.n_points)*1.0e8

    def sample(self) -> ndarray:
        """
        Samples new points from the PSO distribution.

        This method generates new points based on the current positions and
        velocities of the particles, as well as their personal best positions
        and the global best position.

        Returns:
            A NumPy array of shape (n_points, dim) representing the new points.
        """

        if (self.stp == 0):
            self.x = np.random.rand(self.n_points, self.dim)
            self.x = self.xmin + self.x*(self.xmax-self.xmin)
            self.v = np.random.randn(self.n_points, self.dim)*self.v0
        else:
            # Compute global best point
            xb = self.p_best[np.argmin(self.p_score), :]

            # Update
            for i in range(self.n_points):
                r1, r2       = np.random.rand(2)
                self.v[i,:]  = (self.w*self.v[i,:] +
                                self.c1*r1*(self.p_best[i,:] - self.x[i,:]) +
                                self.c2*r2*(xb[:]            - self.x[i,:]))
                self.x[i,:] += self.v[i,:]

        return self.x

    def step(self, x: ndarray, c: ndarray) -> None:
        """
        Performs one step of the PSO algorithm.

        This method updates the local best positions and scores of the particles
        and increments the step counter.

        Args:
            x: The points that were evaluated.
            c: The cost values at the evaluated points.
        """

        # Update best
        self.update_local_best(x, c)

        self.stp += 1

    def update_local_best(self, x: ndarray, c: ndarray) -> None:
        """
        Updates the local best positions and scores of the particles.

        Args:
            x: The points that were evaluated.
            c: The cost values at the evaluated points.
        """

        # Update best score for each particle
        for i in range(self.n_points):
            if (c[i] <= self.p_score[i]):
                self.p_score[i]  = c[i]
                self.p_best[i,:] = x[i,:]
