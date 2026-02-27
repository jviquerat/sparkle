from typing import Tuple

import numpy as np
from numpy import ndarray

from sparkle.src.env.parallel import parallel
from sparkle.src.utils.error import error


###############################################
class BaseParallelEnvironments():
    """
    A base class for parallel environments.

    This class provides a common interface for managing and interacting with
    multiple environments running in parallel. It includes methods for
    evaluating costs, generating cost maps, and handling parallel execution.
    """
    def __init__(self):
        """
        Initializes the BaseParallelEnvironments.
        """

    def evaluate(self, x: ndarray, verbose: bool = True) -> ndarray:
        """
        Evaluates the cost of multiple points in parallel.

        Args:
            x: A NumPy array of points to evaluate.
            verbose: Whether to print progress.

        Returns:
            A NumPy array of the corresponding costs.

        Raises:
            error: If the number of points is not a multiple of the number of parallel environments.
        """

        n_points = x.shape[0]

        if (n_points%parallel.n_envs != 0):
            error("base_parallel_environments", "evaluate",
                  "nb of evaluation pts should be a multiple of nb of parallel envs")

        n_steps = n_points//parallel.n_envs
        costs   = np.zeros(n_points)

        step = 0
        while (step < n_steps):
            end = "\r"
            if (step == n_steps-1): end = "\n"
            i_start = step*parallel.n_envs
            i_end   = (step+1)*parallel.n_envs - 1
            if verbose:
                print("# Computing individuals #"+str(i_start)+" to #"+str(i_end), end=end)

            xp = np.zeros((parallel.n_envs, self.spaces.dim))
            for k in range(parallel.n_envs):
                xp[k,:] = x[step*parallel.n_envs + k]

            c = self.cost(xp)
            for k in range(parallel.n_envs):
                costs[step*parallel.n_envs + k] = c[k]

            step += 1

        return costs

    def generate_cost_map_1D(self) -> Tuple[ndarray, ndarray]:
        """
        Generates a cost map for rendering 1D environments.

        Returns:
            A tuple containing:
                - A NumPy array of x-coordinates.
                - A NumPy array of corresponding cost values.
        """

        n_plot   = 400
        x_plot   = np.linspace(self.spaces.xmin[0], self.spaces.xmax[0], num=n_plot)

        # Flatten all points
        x_flat = np.zeros((n_plot, 1))
        for i in range(n_plot):
            x_flat[i, 0] = x_plot[i]

        # Pad to multiple of parallel.n_envs
        remainder = n_plot % parallel.n_envs
        if remainder != 0:
            pad_size = parallel.n_envs - remainder
            x_flat = np.vstack((x_flat, np.tile(x_flat[-1,:], (pad_size, 1))))

        # Evaluate all points in parallel
        costs_flat = self.evaluate(x_flat, verbose=False)

        # Trim padding
        cost_map = costs_flat[:n_plot]

        return x_plot, cost_map

    def generate_cost_map_2D(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Generates a cost map for rendering 2D environments.

        Returns:
            A tuple containing:
                - A NumPy array of x-coordinates.
                - A NumPy array of y-coordinates.
                - A NumPy array of corresponding cost values.
        """

        n_plot   = 100
        x_plot   = np.linspace(self.spaces.xmin[0], self.spaces.xmax[0], num=n_plot)
        y_plot   = np.linspace(self.spaces.xmax[1], self.spaces.xmin[1], num=n_plot)
        grid     = np.array(np.meshgrid(x_plot, y_plot))
        x_plot   = grid[0]
        y_plot   = grid[1]

        # Flatten all points into an array of shape (N, 2)
        n_points = n_plot * n_plot
        x_flat = np.zeros((n_points, 2))
        for i in range(n_plot):
            for j in range(n_plot):
                x_flat[i*n_plot+j, :] = [x_plot[i,j], y_plot[i,j]]

        # Pad to multiple of parallel.n_envs
        remainder = n_points % parallel.n_envs
        if remainder != 0:
            pad_size = parallel.n_envs - remainder
            x_flat = np.vstack((x_flat, np.tile(x_flat[-1,:], (pad_size, 1))))

        # Evaluate all points in parallel
        costs_flat = self.evaluate(x_flat, verbose=False)

        # Trim padding and reshape back to grid
        cost_map = np.reshape(costs_flat[:n_points], (n_plot, n_plot))

        return x_plot, y_plot, cost_map
