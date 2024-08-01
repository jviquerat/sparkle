# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.pex.base import base_pex

###############################################
### Random experiment plan
class random(base_pex):
    def __init__(self, dim, xmin, xmax, pms):

        self.name_             = "random"
        self.dim_              = dim
        self.xmin_             = xmin
        self.xmax_             = xmax
        self.n_points_         = pms.n_points
        self.n_points_per_dim_ = math.floor(math.pow(self.n_points_, 1.0/self.dim_))

        # Generate x points for pex
        self.x_ = np.random.uniform(low  = self.xmin_,
                                    high = self.xmax_,
                                    size = self.n_points_*self.dim_)
        self.x_ = np.reshape(self.x_, (self.n_points_per_dim_, self.dim_))

