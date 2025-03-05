# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.pex.base      import base_pex
from sparkle.src.utils.error   import error
from sparkle.src.utils.default import set_default
from sparkle.src.utils.prints  import spacer

###############################################
### Poisson-disc experiment plan
### Relies on Robert Bridson algorithm
### XXX 2D only for now
class fixed_poisson_disc(base_pex):
    def __init__(self, spaces, pms):
        super().__init__(spaces)

        self.name_       = "fixed_poisson_disc"
        self.n_points_   = pms.n_points
        self.n_attempts_ = set_default("n_attempts", 20, pms)

        # Compute radius guess
        self.radius_ = 0.8*math.sqrt(self.volume()/self.n_points_)

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
                                       size = self.n_attempts_)
            radius = np.random.uniform(low  = self.radius_,
                                       high = 2.0*self.radius_,
                                       size = self.n_attempts_)

            for i in range(self.n_attempts_):
                x  = active[k][0] + radius[i]*math.cos(theta[i])
                y  = active[k][1] + radius[i]*math.sin(theta[i])
                pt = np.array([x,y])

                if ((pt[0] < self.xmin[0]) or
                    (pt[0] > self.xmax[0]) or
                    (pt[1] < self.xmin[1]) or
                    (pt[1] > self.xmax[1])): continue

                ok = True
                for j in range(len(lst)):
                    dist = self.distance(pt, lst[j])
                    if (dist < self.radius_):
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
            error("poisson_disk", "reset",
                  "the resulting number of points was lower than n_points_")

        # Save initial number of points
        self.n_initial_points = len(lst)

        # Farthest point sampling
        k = np.random.randint(0, len(lst))
        selected = [lst[k]]
        lst.pop(k)

        while (len(selected) < self.n_points_):
            distances = np.zeros((len(lst), len(selected)))
            for i in range(len(lst)):
                for j in range(len(selected)):
                    distances[i,j] = self.distance(lst[i], selected[j])

            min_dists = np.min(distances, axis=1)
            k         = np.argmax(min_dists)

            selected.append(lst[k])
            lst.pop(k)

        self.x_ = np.array(selected)

        # Compute minimal distance
        self.d_min = self.dist_min(self.x)

    # Distance
    def distance(self, p1, p2):

        return np.linalg.norm(p1 - p2, 2)

    # Print informations
    def summary(self):

        super().summary()
        spacer("Initial nb of pts: "+str(self.n_initial_points))
        spacer("Final min distance: "+str(self.d_min))
