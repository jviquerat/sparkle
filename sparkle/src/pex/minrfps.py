from types import SimpleNamespace

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.utils.default import set_default


###############################################
class MinRFPS(BasePex):
    """
    Minimax-oriented Furthest Point Sampling (FPS).

    This class implements a random sampling method followed by a Furthest
    Point Sampling step to improve the distribution of points.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the RandomFPS experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan,
                including the factor for oversampling (factor).
        """
        super().__init__(spaces, pms)

        self.name    = "minrfps"
        self.factor  = set_default("factor", 20, pms)
        self.n_large = self.factor*self.n_points_

        self.reset()

    def reset(self) -> None:
        """
        Resets the MinRFPS experiment plan by generating new sample points.

        This method generates an initial set of random points, then applies
        minimax Furthest Point Sampling to select a subset of these points that are
        well-distributed according to the minimax criterion
        """

        x = np.random.uniform(low  = self.xmin,
                              high = self.xmax,
                              size = (self.n_large, self.dim))

        self.x_ = self.minimax_FPS(x, self.n_points_, 10000)

    def minimax_FPS(self, x: ndarray, n_points: int, n_samples: int) -> ndarray:
        """
        A minimax-oriented furthest point sampling
        Uses an incremental minimax computation function to select
        points from a pool of candidates and add them to the design

        Args:
            x: numpy array of shape (n,d)
            n_points: target number of points
            n_samples: number of samples to use for the MC minimax evaluation

        Returns:
            the selected subset of x
        """
        # selected is the array of selected points indices
        # mask[p] = True if point p is candidate
        n_large  = x.shape[0]
        selected = np.zeros(n_points, dtype=int)
        mask     = np.ones(n_large, dtype=bool)

        # Array of minimal (squared) distance from each point to any selected point
        # It is initialized with np.inf and updated after each point selection
        min_dists = np.full(n_large, np.inf)

        # Select the first point randomly
        k           = np.random.randint(0, n_large)
        selected[0] = k
        mask[k]     = False
        n_selected  = 1

        # Compute (squared) distances from current point to candidates
        diff         = x[mask] - x[k]
        dists_to_crt = np.sum(diff*diff, axis=1)

        # Compute (squared) distances from current point to MC set
        self.MC_set = np.random.uniform(low=self.xmin,
                                        high=self.xmax,
                                        size=(n_samples, self.dim))
        diff = self.MC_set - x[k]
        dists_to_MC_set = np.sum(diff*diff, axis=1)

        # Loop until n_target points are selected
        while n_selected < n_points:

            # Find worst-covered point from MC set
            worst_pt_idx = np.argmax(dists_to_MC_set)
            worst_pt = self.MC_set[worst_pt_idx]

            original_indices = np.where(mask)[0]
            candidate_coords = x[original_indices]

            # Find point in candidates closest to worst covered pt
            diff              = candidate_coords - worst_pt
            dists_to_worst_pt = np.sum(diff**2, axis=1)
            best_idx          = np.argmin(dists_to_worst_pt)

            # Retrieve original index
            best_idx = original_indices[best_idx]

            # Add the farthest point to the selected set
            selected[n_selected] = best_idx
            mask[best_idx]       = False # Mark as selected
            n_selected          += 1

            # Update dists_to_MC_set
            diff = self.MC_set - x[best_idx]
            dist = np.sum(diff*diff, axis=1)
            dists_to_MC_set = np.minimum(dists_to_MC_set, dist)

            # Exit if done
            if n_selected == n_points:
                break

        # Return the subset of points corresponding to the selected indices
        return x[selected]
