from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.fps import FPS
from sparkle.src.utils.default import set_default


###############################################
class Random(BasePex):
    """
    Random experiment plan.

    This class implements a simple random sampling method for generating
    experiment plans. Points are uniformly distributed within the search space.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the Random experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan.
        """
        super().__init__(spaces, pms)

        self.name = "random"

        self.reset()

    def reset(self) -> None:
        """
        Resets the Random experiment plan by generating new sample points.

        This method generates a new set of points randomly distributed
        within the search space.
        """

        self.x_ = np.random.uniform(low  = self.xmin,
                                    high = self.xmax,
                                    size = (self.n_points_, self.dim))
