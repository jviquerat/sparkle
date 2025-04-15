import math
from types import SimpleNamespace

import numpy as np
from numpy import ndarray

from sparkle.src.agent.base import BaseAgent
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default
from sparkle.src.utils.error import error

###############################################
class SA(BaseAgent):
    """
    Simulated Annealing (SA) agent

    This agent implements the Simulated Annealing algorithm, a probabilistic
    technique for approximating the global optimum of a given function.
    It explores the search space by accepting worse solutions with a
    probability that decreases over time (as temperature cools)
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: SimpleNamespace) -> None:
        """
        Initializes the SA agent.

        Args:
            path: The base path for storing results
            spaces: The environment's search space definition
            pms: A SimpleNamespace object containing parameters for the agent
                 Expected parameters:
                 - n_steps_max (int): Maximum number of iterations (steps)
                 - T0 (float): Initial temperature
                 - alpha (float): Cooling rate (e.g., 0.95). Must be < 1
                 - std_factor (float, optional): Factor to determine
                   standard deviation for neighbor generation relative to search
                   space range (default: 0.05)
                 - silent (bool, optional): Suppress summary output
        """
        super().__init__(path, spaces, pms)

        self.name        = "SA"
        self.n_points    = 1
        self.n_steps_max = set_default("n_steps_max", 100, pms)
        self.T0          = set_default("T0", 1.0, pms)
        self.alpha       = set_default("alpha", 0.95, pms)
        self.std_factor  = set_default("std_factor", 0.05, pms)

        if not (0 < self.alpha < 1):
            error("SA", "__init__", "alpha must be between 0 and 1")
        if self.T0 <= 0:
            error("SA", "__init__", "T0 must be positive")

        # Compute standard deviation for neighbor generation
        self.neighbor_std = (self.xmax - self.xmin) * self.std_factor

        if (not self.silent): self.summary()

    def reset(self, run: int) -> None:
        """
        Resets the SA agent for a new run

        Args:
            run: The run number
        """
        super().reset(run)

        self.T = self.T0
        self.x_current = self.x0.copy()
        self.c_current = np.inf # Will be updated after the first step

        self.x_best = self.x_current.copy()
        self.c_best = np.inf

    def sample(self) -> ndarray:
        """
        Samples a new point for evaluation

        For the first step, it returns the initial point (x0)
        For subsequent steps, it generates a neighbor of the current point

        Returns:
            A NumPy array of shape (1, dim) representing the point to evaluate
        """
        if self.stp == 0:
            # Return the initial point for the first evaluation
            point_to_sample = self.x_current
        else:
            # Generate a neighboring point
            noise = np.random.randn(self.dim) * self.neighbor_std
            x_neighbor = self.x_current + noise

            # Ensure the neighbor stays within bounds
            point_to_sample = np.clip(x_neighbor, self.xmin, self.xmax)

        # Return as shape (1, dim) because n_points is 1
        return point_to_sample.reshape(1, self.dim)

    def step(self, x: ndarray, c: ndarray) -> None:
        """
        Performs one step of the SA algorithm

        Updates the current state based on the acceptance probability
        of the evaluated point, updates the best solution found,
        cools the temperature, and increments the step counter.

        Args:
            x: The point(s) that were evaluated (shape (1, dim))
            c: The cost value(s) at the evaluated points (shape (1,))
        """
        # Since n_points = 1, x has shape (1, dim) and c has shape (1,)
        x_eval = x[0]
        c_eval = c[0]

        if self.stp == 0:
            # This was the evaluation of the initial point
            self.x_current = x_eval
            self.c_current = c_eval
            self.x_best = x_eval.copy()
            self.c_best = c_eval
        else:
            # This was the evaluation of a neighbor point
            delta_c = c_eval - self.c_current

            # Decide whether to accept the move (assuming minimization)
            accept = False
            if delta_c < 0:
                accept = True
            else:
                # Avoid division by zero or math domain error if T is very small
                if self.T > 1e-9:
                    prob = np.exp(-delta_c / self.T)
                    if np.random.rand() < prob:
                        accept = True

            if accept:
                self.x_current = x_eval
                self.c_current = c_eval

            # Update the overall best solution found
            if self.c_current < self.c_best:
                self.x_best = self.x_current.copy()
                self.c_best = self.c_current

            # Cool down the temperature
            self.T = max(self.T * self.alpha, 1e-9) # Geometric cooling, prevent T=0

        self.stp += 1
