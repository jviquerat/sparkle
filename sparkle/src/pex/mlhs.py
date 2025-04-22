import math
from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.default import set_default
from sparkle.src.utils.distances import (nearest_neighbor_in_set, nearest_neighbors_in_set,
                                         pairwise_distances)
from sparkle.src.utils.prints import fmt_float, spacer


class MLHS(BasePex):
    """
    Maximin Latin Hypercube Sampling (MLHS) experiment plan

    Combines LHS stratification with a maximin criterion (maximizing the minimum
    distance between points) for improved space-filling properties
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the MLHS experiment plan

        Args:
            spaces: The environment's search space definition
            pms: A SimpleNamespace object containing other parameters
        """
        super().__init__(spaces, pms)

        self.name = "maximin_lhs"

        default_swap_ratio = max(0.1, 1.0/float(self.dim))
        self.swap_ratio    = set_default("swap_ratio", default_swap_ratio, pms)
        default_n_iter     = math.ceil(self.swap_ratio*self.dim*self.n_points_)
        self.n_iter        = set_default("n_iter", default_n_iter, pms)
        self.pms           = pms

        self.reset()

    def acceptance(self, new_min_dist: float) -> bool:
        """
        An acceptance function for the standard MLHS design
        If the new minimal distance is higher than the previous one, accept the swap

        Args:
            new_min_dist: new minimal distance in design

        Returns:
            accepted: whether to accept the swap
        """

        accepted = False
        if new_min_dist > self.d_min:
            accepted = True

        return accepted

    def reset(self) -> None:
        """
        Resets the MLHS plan by generating and optimizing sample points
        """

        # Generate initial LHS design
        initial_design = LHS(self.spaces, self.pms)
        self.x_ = initial_design.x.copy()

        # Compute initial nearest neighbors
        d_nearest, p_nearest = nearest_neighbors_in_set(self.x)
        p_min_idx            = np.argmin(d_nearest)
        self.d_min           = d_nearest[p_min_idx]
        self.d_min_initial   = self.d_min
        self.n_swaps         = 0

        # Store the current best state (nearest distances and their indices)
        current_d_nearest = d_nearest.copy()
        current_p_nearest = p_nearest.copy()

        # Track the best design found so far
        best_x = self.x_.copy()
        best_d_min = self.d_min

        # Loop trying to space out samples
        for iteration in range(self.n_iter):

            # Choose a random dimension to perform the swap
            dim_idx = np.random.randint(self.dim)

            # Choose two distinct random points from the design to swap coordinates
            p1, p2 = np.random.choice(self.n_points, size=2, replace=False)

            # Store original coordinate values before swapping
            x1_dim_val = self.x[p1, dim_idx]
            x2_dim_val = self.x[p2, dim_idx]

            # Perform the swap on the design matrix
            self.x[p1, dim_idx] = x2_dim_val
            self.x[p2, dim_idx] = x1_dim_val

            # Create temporary copies to hold the state after the potential swap
            temp_d_nearest = current_d_nearest.copy()
            temp_p_nearest = current_p_nearest.copy()

            # Update nearest neighbor info for the two swapped points (p1, p2)
            dn1, pn1 = nearest_neighbor_in_set(self.x, p1)
            dn2, pn2 = nearest_neighbor_in_set(self.x, p2)
            temp_d_nearest[p1], temp_p_nearest[p1] = dn1, pn1
            temp_d_nearest[p2], temp_p_nearest[p2] = dn2, pn2

            # Update nearest neighbor info for points whose nearest neighbor
            # was p1 or p2 before the swap. First we identify these points,
            # then we recompute the nearest neighbors
            affected_indices = np.where((current_p_nearest == p1) |
                                        (current_p_nearest == p2))[0]

            for k in affected_indices:
                # If we find p1 or p2, this was already done in previous step
                if k == p1 or k == p2: continue

                dnk, pnk = nearest_neighbor_in_set(self.x, k)
                temp_d_nearest[k], temp_p_nearest[k] = dnk, pnk

            # For all points, check if the new p1 or p2 are now nearest neighbors
            # We first identify these points, then check for nearest neighbors
            other_indices_mask = np.ones(self.n_points, dtype=bool)
            other_indices_mask[[p1, p2]] = False # exclude p1 and p2
            other_indices_mask[affected_indices] = False # exclude pts from previous case
            other_indices = np.where(other_indices_mask)[0]

            if len(other_indices) > 0:
                # Compute distances from other points to the modified p1 and p2
                dist_matrix_k_p1p2 = pairwise_distances(self.x[other_indices],
                                                        self.x[[p1, p2]])
                dist_k_p1 = dist_matrix_k_p1p2[:, 0] # Distances to p1
                dist_k_p2 = dist_matrix_k_p1p2[:, 1] # Distances to p2

                # Compare and update if the new p1 or p2 is closer
                for i, k in enumerate(other_indices):
                    # Check if p1 is closer than the current nearest
                    p, d = p1, dist_k_p1[i]
                    if (dist_k_p2[i] < dist_k_p1[i]):
                        p, d = p2, dist_k_p2[i]

                    if d < temp_d_nearest[k]:
                        temp_d_nearest[k] = d
                        temp_p_nearest[k] = p

            # Find new min distance and update if improved
            new_min_dist = np.min(temp_d_nearest)

            # Acceptance logic
            accepted = self.acceptance(new_min_dist)

            # Update state based on acceptance
            if accepted: # accept swap: self.x is already modified
                self.d_min = new_min_dist
                current_d_nearest[:] = temp_d_nearest
                current_p_nearest[:] = temp_p_nearest
                self.n_swaps += 1

                # Check if this accepted state is the new best overall
                if self.d_min > best_d_min:
                    best_d_min = self.d_min
                    best_x = self.x.copy()
            else:
                # Reject swap: revert self.x_ to its state before the swap
                self.x[p1, dim_idx] = x1_dim_val
                self.x[p2, dim_idx] = x2_dim_val

        # Set the final design to the best one found during the search
        self.x_ = best_x
        self.d_min = best_d_min

    def summary(self):
        """
        Prints a summary of the MLHS experiment plan's configuration.
        """

        super().summary()
        spacer("Initial min distance: "+fmt_float(self.d_min_initial))
        spacer("Final min distance: "+fmt_float(self.d_min))
        spacer("Total nb of attempted swaps: "+str(self.n_iter))
        spacer("Number of accepted swaps: "+str(self.n_swaps))
