# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.utils.default import set_default
from sparkle.src.agent.base    import base_agent

###############################################
### CEM
class cem(base_agent):
    def __init__(self, path, spaces, pms):
        super().__init__(path, spaces, pms)

        self.name        = "CEM"
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_points    = set_default("n_points", 2*self.dim, pms)
        self.n_elites    = set_default("n_elites", math.floor(self.n_points/2), pms)
        self.alpha       = set_default("alpha", 0.2, pms)

        self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Min and max arrays used for cem adaptation
        self.xmin_cem = self.xmin.copy()
        self.xmax_cem = self.xmax.copy()

    # Sample from distribution
    def sample(self):

        x = np.zeros((self.n_points, self.dim))

        for i in range(self.n_points):
            x[i,:] = np.random.uniform(low  = self.xmin_cem,
                                       high = self.xmax_cem)

        return x

    # Step
    def step(self, x, c):

        # Sort
        self.sort(x, c)

        # Update xmin and xmax
        xmin = np.amin(x[:self.n_elites,:], axis=0)
        xmax = np.amax(x[:self.n_elites,:], axis=0)
        self.xmin_cem[:] = ((1.0-self.alpha)*self.xmin_cem[:] + self.alpha*xmin[:])
        self.xmax_cem[:] = ((1.0-self.alpha)*self.xmax_cem[:] + self.alpha*xmax[:])

        self.stp += 1

    # Sort offsprings based on cost
    # x and c arrays are actually modified here
    def sort(self, x, c):

        sc   = np.argsort(c)
        x[:] = x[sc[:]]
        c[:] = c[sc[:]]
