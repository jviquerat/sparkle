from types import SimpleNamespace
from typing import Generator

import numpy as np

from sparkle.src.agent.base import BaseAgent
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default


###############################################
class NelderMead(BaseAgent):
    """
    This agent implements the Nelder-Mead simplex algorithm
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: SimpleNamespace) -> None:
        """
        Initializes the Nelder-Mead agent

        Args:
            path: The base path for storing results
            spaces: The environment's search space definition
            pms: A SimpleNamespace object containing parameters for the agent
                 - n_steps_max (int): Maximum number of evaluations (default: 1000)
                 - alpha (float): Reflection coefficient (default: 1.0)
                 - gamma (float): Expansion coefficient (default: 2.0)
                 - rho (float): Contraction coefficient (default: 0.5)
                 - sigma (float): Shrink coefficient (default: 0.5)
                 - delta (float): Step size for initial simplex (default: 0.05)
        """
        super().__init__(path, spaces, pms)

        self.name        = "NelderMead"
        self.n_points    = 1 # We evaluate one point at a time mostly

        self.n_steps_max = set_default("n_steps_max", 1000, pms)
        self.alpha       = set_default("alpha", 1.0, pms)
        self.gamma       = set_default("gamma", 2.0, pms)
        self.rho         = set_default("rho", 0.5, pms)
        self.sigma       = set_default("sigma", 0.5, pms)
        self.delta       = set_default("delta", 0.05, pms)

        if (not self.silent): self.summary()

    def reset(self, run: int) -> None:
        """
        Resets the agent for a new run
        """
        super().reset(run)

        # Initialize generator
        self.gen = self._algorithm()
        self.next_point = next(self.gen)

    def sample(self, validate) -> np.ndarray:
        """
        Samples a new point for evaluation
        """
        return self.next_point.reshape(1, self.dim)

    def step(self, x: np.ndarray, c: np.ndarray) -> None:
        """
        Performs one step of the algorithm
        """
        self.next_point = self.gen.send(c[0])
        self.stp += 1

    def _algorithm(self) -> Generator[np.ndarray, float, None]:
        """
        Nelder-Mead algorithm logic as a generator

        Yields:
            The next point to be evaluated (as a numpy array)
        Receives:
            The cost of the yielded point (via .send())
        """

        # Initialization
        simplex_x = np.zeros((self.dim + 1, self.dim))
        simplex_f = np.zeros(self.dim + 1)

        # Evaluate x0
        simplex_x[0] = self.x0.copy()
        simplex_f[0] = yield simplex_x[0]

        # Evaluate remaining N points (x0 + delta * e_i)
        for i in range(self.dim):
            point = simplex_x[0].copy()
            step = self.delta * (self.xmax[i] - self.xmin[i])
            point[i] += step

            # Check bounds (simple reflection if out of bounds)
            if point[i] > self.xmax[i]:
                point[i] -= 2.0*step

            point = np.clip(point, self.xmin, self.xmax)
            simplex_x[i+1] = point
            simplex_f[i+1] = yield point

        # Main loop
        while True:
            # Sort indices based on cost
            order = np.argsort(simplex_f)
            simplex_x = simplex_x[order]
            simplex_f = simplex_f[order]

            # Best, worst, second worst
            f_best = simplex_f[0]
            f_worst = simplex_f[-1]
            f_second_worst = simplex_f[-2]

            # Centroid of the best N points (all except worst)
            x_centroid = np.mean(simplex_x[:-1], axis=0)

            # Reflection step
            x_reflect = x_centroid + self.alpha*(x_centroid - simplex_x[-1])
            x_reflect = np.clip(x_reflect, self.xmin, self.xmax)
            f_reflect = yield x_reflect

            if f_best <= f_reflect < f_second_worst:
                # Accept Reflection
                simplex_x[-1] = x_reflect
                simplex_f[-1] = f_reflect

            elif f_reflect < f_best:
                # Expansion
                x_expand = x_centroid + self.gamma*(x_reflect - x_centroid)
                x_expand = np.clip(x_expand, self.xmin, self.xmax)
                f_expand = yield x_expand

                if f_expand < f_reflect:
                    simplex_x[-1] = x_expand
                    simplex_f[-1] = f_expand
                else:
                    simplex_x[-1] = x_reflect
                    simplex_f[-1] = f_reflect

            else: # f_reflect >= f_second_worst
                # Contraction
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = x_centroid + self.rho*(x_reflect - x_centroid)
                    x_contract = np.clip(x_contract, self.xmin, self.xmax)
                    f_contract = yield x_contract

                    if f_contract <= f_reflect:
                        simplex_x[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        simplex_x, simplex_f = (yield from self._shrink(simplex_x, simplex_f))

                else: # f_reflect >= f_worst
                    # Inside contraction
                    x_contract = x_centroid + self.rho*(simplex_x[-1] - x_centroid)
                    x_contract = np.clip(x_contract, self.xmin, self.xmax)
                    f_contract = yield x_contract

                    if f_contract < simplex_f[-1]:
                        simplex_x[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        simplex_x, simplex_f = (yield from self._shrink(simplex_x, simplex_f))

    def _shrink(self, simplex_x, simplex_f):
        """
        Performs the shrink operation
        Yields from generator to evaluate points
        Returns updated simplex
        """
        x_best = simplex_x[0]
        # Evaluate new points for all except the best
        for i in range(1, self.dim + 1):
            simplex_x[i] = x_best + self.sigma * (simplex_x[i] - x_best)
            simplex_x[i] = np.clip(simplex_x[i], self.xmin, self.xmax)
            simplex_f[i] = yield simplex_x[i]

        return simplex_x, simplex_f
