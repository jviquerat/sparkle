# Generic imports
import numpy as np

# Custom imports
from sparkle.src.pex.base      import BasePex
from sparkle.src.pex.fps       import FPS
from sparkle.src.utils.default import set_default

###############################################
### Random experiment plan
class Random(BasePex):
    def __init__(self, spaces, pms):
        super().__init__(spaces, pms)

        self.name = "random"

        self.reset()

    # Reset sampling
    def reset(self):

        self.x_ = np.random.uniform(low  = self.xmin,
                                    high = self.xmax,
                                    size = (self.n_points_, self.dim))

###############################################
### Random experiment plan with fps step
class RandomFPS(BasePex):
    def __init__(self, spaces, pms):
        super().__init__(spaces, pms)

        self.name     = "random_fps"
        self.factor   = set_default("factor", 5, pms)
        self.n_target = self.factor*self.n_points_

        self.reset()

    # Reset sampling
    def reset(self):

        x = np.random.uniform(low  = self.xmin,
                              high = self.xmax,
                              size = (self.n_target, self.dim))

        self.x_ = FPS(x, self.n_points_)
