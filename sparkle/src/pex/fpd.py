import math
from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.fps import FPS
from sparkle.src.utils.default import set_default
from sparkle.src.utils.distances import distance, min_distance_in_set
from sparkle.src.utils.error import error
from sparkle.src.utils.prints import fmt_float, spacer


###############################################
class FPD(BasePex):
    """
    Fixed Poisson-Disc (FPD) experiment plan.

    This class implements the Fixed Poisson-Disc sampling method for generating
    experiment plans. FPD ensures that points are well-spaced and avoid
    clustering, providing good coverage of the search space. Relies on Robert Bridson algorithm
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the FPD experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan,
                including the number of attempts (n_attempts).
        """
        super().__init__(spaces, pms)

        self.name       = "fixed_poisson_disc"
        self.n_attempts = set_default("n_attempts", 20, pms)

        # Radius estimate based on volume
        self.radius = 0.75*math.pow(self.volume()/self.n_points_, 1.0/self.dim)

        # Radius estimate from:
        # Sample elimination for generating poisson disk sample sets, C. Yuksel (2015)
        # if (self.dim == 2): self.radius = math.pow(self.volume()/(2.0*math.sqrt(3.0)*self.n_points_), 1.0/2.0)
        # if (self.dim == 3): self.radius = math.pow(self.volume()/(4.0*math.sqrt(2.0)*self.n_points_), 1.0/3.0)
        # if (self.dim >  3):
        #     if (self.dim%2 == 0):
        #         d_start = 4
        #         C       = math.pi
        #     else:
        #         d_start = 3
        #         C       = 1.0
        #     for d in range(d_start, self.dim, 2): C *= 2.0*math.pi/float(d)
        #     self.radius = math.pow(self.volume()/(C*self.n_points_), 1.0/float(self.dim))

        self.reset()

    def reset(self) -> None:
        """
        Resets the FPD experiment plan by generating new sample points.

        This method generates a new set of points using the Poisson-Disc
        sampling algorithm, followed by a Furthest Point Sampling (FPS) step
        to ensure the correct number of points.
        """

        # Poisson-disc sampling
        p = np.random.uniform(low  = self.xmin,
                              high = self.xmax,
                              size = self.dim)

        lst    = [p]
        active = [p]

        while (len(active) > 0):
            found      = False
            k          = np.random.randint(0, len(active))
            radius     = np.random.uniform(low  = self.radius,
                                           high = 2.0*self.radius,
                                           size = self.n_attempts)
            direction  = np.random.normal(size=(self.n_attempts,self.dim))
            direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]

            for i in range(self.n_attempts):
                pt = active[k] + radius[i]*direction[i]
                if not (np.all(self.xmin <= pt) and np.all(pt <= self.xmax)): continue

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
        self.x_ = FPS(np.array(lst), self.n_points_)

        # Compute minimal distance
        self.d_min = min_distance_in_set(self.x)

    def summary(self):
        """
        Prints a summary of the FPD experiment plan's configuration.
        """

        super().summary()
        spacer("Initial nb of pts: "+str(self.n_initial_points))
        spacer("Final min distance: "+fmt_float(self.d_min))
