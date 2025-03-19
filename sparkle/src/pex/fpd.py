# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.pex.base        import base_pex
from sparkle.src.pex.fps         import fps
from sparkle.src.utils.distances import distance, min_distance
from sparkle.src.utils.error     import error
from sparkle.src.utils.default   import set_default
from sparkle.src.utils.prints    import spacer, fmt_float

###############################################
### Fixed poisson-disc experiment plan
### Relies on Robert Bridson algorithm
### XXX 2D only for now
class fpd(base_pex):
    def __init__(self, spaces, pms):
        super().__init__(spaces, pms)

        self.name       = "fixed_poisson_disc"
        self.n_attempts = set_default("n_attempts", 20, pms)

        # Compute radius guess
        self.radius = 0.75*math.sqrt(self.volume()/self.n_points_)

        self.reset()

    # Reset sampling
    # We start by generating a fine poisson-disc sampling, then
    # apply a furthest point sampling on the resulting set to
    # ensure that we have exactly n_points_
    def reset(self):

        # Poisson-disc sampling
        p = np.random.uniform(low  = self.xmin,
                              high = self.xmax,
                              size = self.dim)

        lst    = [p]
        active = [p]

        while (len(active) > 0):
            found  = False
            k      = np.random.randint(0, len(active))
            theta  = np.random.uniform(low  = 0.0,
                                       high = 2.0*math.pi,
                                       size = self.n_attempts)
            radius = np.random.uniform(low  = self.radius,
                                       high = 2.0*self.radius,
                                       size = self.n_attempts)

            for i in range(self.n_attempts):
                x  = active[k][0] + radius[i]*math.cos(theta[i])
                y  = active[k][1] + radius[i]*math.sin(theta[i])
                pt = np.array([x,y])

                if ((pt[0] < self.xmin[0]) or
                    (pt[0] > self.xmax[0]) or
                    (pt[1] < self.xmin[1]) or
                    (pt[1] > self.xmax[1])): continue

                ok = True
                for j in range(len(lst)):
                    dist = distance(pt, lst[j])
                    if (dist < self.radius):
                        ok = False
                        break

                if ok:
                    lst.append(pt)
                    active.append(pt)
                    found = True

            if not found:
                active.pop(k)

        # Check that we have more than n_points_
        if (len(lst) < self.n_points_):
            error("fpd", "reset",
                  "the resulting number of points was lower than n_points_")

        # Save initial number of points
        self.n_initial_points = len(lst)

        # Farthest point sampling
        self.x_ = fps(np.array(lst), self.n_points_)

        # Compute minimal distance
        self.d_min = min_distance(self.x)

    # Print informations
    def summary(self):

        super().summary()
        spacer("Initial nb of pts: "+str(self.n_initial_points))
        spacer("Final min distance: "+fmt_float(self.d_min))
