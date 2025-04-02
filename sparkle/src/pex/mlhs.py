import math
from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.default import set_default
from sparkle.src.utils.distances import distance, nearest_all_to_all, nearest_one_to_all
from sparkle.src.utils.prints import fmt_float, spacer


###############################################
class MLHS(BasePex):
    """
    Maximin Latin Hypercube Sampling (MLHS) experiment plan.

    This class implements the Maximin Latin Hypercube Sampling method, which
    combines the stratification of LHS with a maximin criterion to improve
    the space-filling properties of the experiment plan.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the MLHS experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan,
                including the swap ratio (swap_ratio).
        """
        super().__init__(spaces, pms)

        self.name       = "maximin_lhs"
        self.swap_ratio = set_default("swap_ratio", max(0.1, 1.0/float(self.dim)), pms)
        self.n_iter     = math.floor(self.swap_ratio*self.dim*self.n_points_**2)
        self.pms        = pms

        self.reset()

    def reset(self) -> None:
        """
        Resets the MLHS experiment plan by generating new sample points.

        This method generates an initial LHS design, then iteratively
        improves it by swapping points to maximize the minimum distance
        between points.
        """

        # Generate default lhs
        base    = LHS(self.spaces, self.pms)
        self.x_ = base.x

        # Compute initial nearest neighbors
        d_nearest, p_nearest = nearest_all_to_all(self.x)
        p_min                = np.argmin(d_nearest)
        self.d_min_initial   = d_nearest[p_min]
        self.d_min           = self.d_min_initial
        self.n_swaps         = 0

        # Copy nearest arrays
        dn_copy = d_nearest.copy()
        pn_copy = p_nearest.copy()

        # Space out samples
        for k in range(self.n_iter):

            # Draw a random dimension
            dim = np.random.randint(self.dim)

            # Draw two distinct random points from the samples
            p1, p2 = np.random.choice(self.n_points, size=2, replace=False)

            # Save original p1 and p2 coordinates
            x1 = self.x[p1].copy()
            x2 = self.x[p2].copy()

            # Exchance dimension dim between points p1 and p2
            self.x[p1,dim] = x2[dim]
            self.x[p2,dim] = x1[dim]

            # Update nearest for pts that had p1 or p2 as nearest
            for k in range(self.n_points):
                if (pn_copy[k] in [p1, p2]):
                    dn, pn     = nearest_one_to_all(self.x, k)
                    dn_copy[k] = dn
                    pn_copy[k] = pn

            # Update nearest for p1
            dn, pn      = nearest_one_to_all(self.x, p1)
            dn_copy[p1] = dn
            pn_copy[p1] = pn

            # Update nearest for p2
            dn, pn      = nearest_one_to_all(self.x, p2)
            dn_copy[p2] = dn
            pn_copy[p2] = pn

            # For all points, check if p1 or p2 is now nearest
            for k in range(self.n_points):
                if (k == p1) or (k == p2): continue
                d1 = distance(self.x[k], self.x[p1])
                d2 = distance(self.x[k], self.x[p2])
                p, d = p1, d1
                if (d2 < d1): p, d = p2, d2
                if (d < dn_copy[k]):
                    dn_copy[k] = d
                    pn_copy[k] = p

            # Compute new min distance and update if improved
            p = np.argmin(dn_copy)
            d = dn_copy[p]
            if (d > self.d_min):
                self.d_min    = d
                self.n_swaps += 1
                d_nearest[:]  = dn_copy[:]
                p_nearest[:]  = pn_copy[:]
            else:
                self.x[p1,dim] = x1[dim]
                self.x[p2,dim] = x2[dim]
                dn_copy[:]     = d_nearest[:]
                pn_copy[:]     = p_nearest[:]

    def summary(self):
        """
        Prints a summary of the MLHS experiment plan's configuration.
        """

        super().summary()
        spacer("Initial min distance: "+fmt_float(self.d_min_initial))
        spacer("Final min distance: "+fmt_float(self.d_min))
        spacer("Total nb of attempted swaps: "+str(self.n_iter))
        spacer("Number of accepted swaps: "+str(self.n_swaps))
