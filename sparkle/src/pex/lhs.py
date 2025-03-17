# Generic imports
import numpy as np

# Custom imports
from sparkle.src.pex.base import base_pex

###############################################
### Latin hypercube sampling
class lhs(base_pex):
    def __init__(self, spaces, pms):
        super().__init__(spaces)

        self.name      = "lhs"
        self.n_points_ = pms.n_points

        self.reset()

    # Reset sampling
    def reset(self):

        # Generate x points for pex
        low  = np.arange(0,self.n_points_)/self.n_points_
        high = np.arange(1,self.n_points_+1)/self.n_points_

        self.x_ = np.random.uniform(low=low,
                                    high=high,
                                    size=[self.dim,self.n_points_]).T

        for d in range(1,self.dim):
            np.random.shuffle(self.x_[:,d])

        for i in range(self.n_points_):
            self.x_[i] = self.xmin + self.x_[i]*(self.xmax - self.xmin)
