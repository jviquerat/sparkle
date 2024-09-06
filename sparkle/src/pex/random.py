# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.pex.base import base_pex

###############################################
### Random experiment plan
class random(base_pex):
    def __init__(self, dim, xmin, xmax, pms):

        self.name_ = "random"
        self.reset(dim, xmin, xmax, pms)

    # Reset sampling
    def reset(self, dim, xmin, xmax, pms):

        self.dim_              = dim
        self.xmin_             = xmin
        self.xmax_             = xmax
        self.n_points_         = pms.n_points

        # Generate x points for pex
        self.x_ = np.random.uniform(low  = self.xmin_,
                                    high = self.xmax_,
                                    size = (self.n_points_,self.dim_))

