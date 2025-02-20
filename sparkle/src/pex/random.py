# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.pex.base import base_pex

###############################################
### Random experiment plan
class random(base_pex):
    def __init__(self, spaces, pms):
        super().__init__(spaces)

        self.name_     = "random"
        self.n_points_ = pms.n_points

        self.reset()

    # Reset sampling
    def reset(self):

        # Generate x points for pex
        self.x_ = np.random.uniform(low  = self.xmin,
                                    high = self.xmax,
                                    size = (self.n_points_,self.dim))

