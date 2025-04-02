from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex


###############################################
### Latin hypercube sampling
class LHS(BasePex):
    """
    Latin Hypercube Sampling (LHS) experiment plan.

    This class implements the Latin Hypercube Sampling method for generating
    experiment plans. LHS is a stratified sampling technique that ensures
    good coverage of the search space.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the LHS experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan.
        """
        super().__init__(spaces, pms)

        self.name = "lhs"

        self.reset()

    # Reset sampling
    def reset(self) -> None:
        """
        Resets the LHS experiment plan by generating new sample points.

        This method generates a new set of points using the LHS algorithm.
        """

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
