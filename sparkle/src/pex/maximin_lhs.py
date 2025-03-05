# Generic imports
import numpy as np

# Custom imports
from sparkle.src.pex.base      import base_pex
from sparkle.src.pex.lhs       import lhs
from sparkle.src.utils.default import set_default
from sparkle.src.utils.prints  import spacer

###############################################
### Maximin Latin hypercube sampling
class maximin_lhs(base_pex):
    def __init__(self, spaces, pms):
        super().__init__(spaces)

        self.name_     = "maximin_lhs"
        self.n_points_ = pms.n_points
        self.n_iter    = set_default("n_iter", 10000, pms)
        self.pms       = pms

        self.reset()

    # Reset sampling
    def reset(self):

        # Generate default lhs
        base    = lhs(self.spaces, self.pms)
        self.x_ = base.x

        # Save initial min distance
        self.d_min_initial = self.dist_min(self.x)
        self.n_swaps       = 0

        # Space out samples
        self.d_min = self.d_min_initial
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

            # Compute new min distance and update if improved
            d = self.dist_min(self.x)
            if (d > self.d_min):
                self.d_min    = d
                self.n_swaps += 1
            else:
                self.x[p1,dim] = x1[dim]
                self.x[p2,dim] = x2[dim]

    # Print informations
    def summary(self):

        super().summary()
        spacer("Initial min distance: "+str(self.d_min_initial))
        spacer("Final min distance: "+str(self.d_min))
        spacer("Number of accepted swaps: "+str(self.n_swaps))
