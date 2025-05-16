from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.fps import FPS
from sparkle.src.utils.default import set_default


###############################################
class MaxRFPS(BasePex):
    """
    Maximin-oriented Furthest Point Sampling (FPS).

    This class implements a random sampling method followed by a Furthest
    Point Sampling step to improve the distribution of points.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the RandomFPS experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan,
                including the factor for oversampling (factor).
        """
        super().__init__(spaces, pms)

        self.name    = "maxrfps"
        self.factor  = set_default("factor", 5, pms)
        self.n_large = self.factor*self.n_points_

        self.reset()

    def reset(self) -> None:
        """
        Resets the MaxRFPS experiment plan by generating new sample points.

        This method generates an initial set of random points, then applies
        Furthest Point Sampling to select a subset of these points that are
        well-distributed according to the maximin criterion
        """

        x = np.random.uniform(low  = self.xmin,
                              high = self.xmax,
                              size = (self.n_large, self.dim))

        self.x_ = FPS(x, self.n_points_)
