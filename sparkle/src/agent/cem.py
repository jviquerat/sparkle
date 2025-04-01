# Generic imports
import types
import math
import numpy as np

# Custom imports
from sparkle.src.utils.default import set_default
from sparkle.src.agent.base    import BaseAgent
from sparkle.src.pex.mlhs      import MLHS
from sparkle.src.env.spaces    import EnvSpaces

###############################################
### CEM
class CEM(BaseAgent):
    def __init__(self, path, spaces, pms):
        super().__init__(path, spaces, pms)

        self.name        = "CEM"
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_points    = set_default("n_points", 2*self.dim, pms)
        self.n_elites    = set_default("n_elites", math.floor(self.n_points/2), pms)
        self.alpha       = set_default("alpha", 0.2, pms)

        if (not self.silent): self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Min and max arrays used for cem adaptation
        self.xmin_cem = self.xmin.copy()
        self.xmax_cem = self.xmax.copy()

    # Sample from distribution
    def sample(self):
        pms          = types.SimpleNamespace()
        pms.n_points = self.n_points
        pms.n_iter   = 1000

        spaces = {"dim": self.spaces.dim,
                  "xmin": self.xmin_cem,
                  "xmax": self.xmax_cem}

        spaces = EnvSpaces(spaces)
        pex    = MLHS(spaces, pms)

        return pex.x

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
