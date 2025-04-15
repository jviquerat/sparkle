import math
import random
from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.mlhs import LHS, MLHS
from sparkle.src.utils.default import set_default
from sparkle.src.utils.distances import (distance, nearest_neighbor_in_set,
                                         nearest_neighbors_in_set,
                                         pairwise_distances)
from sparkle.src.utils.prints import fmt_float, spacer


class MLHS_SA(MLHS, BasePex):
    """
    Maximin Latin Hypercube Sampling (MLHS) using Simulated Annealing (SA).

    Generates an initial LHS design and iteratively improves it using SA
    to maximize the minimum distance between points. Swaps coordinate
    values between points along one dimension at a time.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the MLHS experiment plan with SA parameters

        Args:
            spaces: The environment's search space definition
            pms: A SimpleNamespace object containing parameters
                 Expected SA parameters in pms:
                 - alpha (float, optional): Cooling factor
                 - n_iter (int, optional): Max iterations. If None, defaults
                   based on the original MLHS formula (ceil(swap_ratio*k*n^2))
                 - swap_ratio (float, optional): Used only if n_iter is None
        """
        BasePex.__init__(self, spaces, pms)

        self.name = "maximin_lhs_sa"
        self.pms = pms

        # Allow explicit n_iter override, otherwise use previous formula
        default_swap_ratio = max(0.1, 1.0/float(self.dim))
        self.swap_ratio    = set_default("swap_ratio", default_swap_ratio, pms)
        default_n_iter     = math.ceil(self.swap_ratio * self.dim * self.n_points_)
        self.n_iter        = set_default("n_iter", default_n_iter, pms)
        self.alpha         = set_default("alpha", 0.98, pms)

        if not (0 < self.alpha < 1):
             error("MLHS_SA", "__init__", "alpha must be between 0 and 1")

        self.T = None

        self.reset() # Implemented in MLHS class

    def acceptance(self, new_min_dist: float) -> bool:
        """
        An acceptance function for the SA-based MLHS design
        The swap accepted probabilistically based on improvement and a random draw

        Args:
            new_min_dist: new minimal distance in design

        Returns:
            accepted: whether to accept the swap
        """
        # On first call, initialize temperature
        if self.T is None:
            self.T = 0.05*self.d_min_initial
            if self.T < 1e-9: self.T = 0.01
        # On following calls, cool down with geometric cooling
        else:
            self.T = max(self.T*self.alpha, 1e-9)

        # Acceptance Logic
        delta_d = new_min_dist - self.d_min # Change in objective (higher is better)
        accepted = False

        if delta_d > 0: # Always accept improvement
            accepted = True
        elif self.T > 1e-9: # Avoid division by zero
            # Accept worse solution probabilistically
            acceptance_prob = math.exp(delta_d / self.T)
            if random.random() < acceptance_prob:
                accepted = True

        return accepted
