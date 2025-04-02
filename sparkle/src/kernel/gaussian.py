from types import SimpleNamespace
from typing import Optional

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.base import BaseKernel
from sparkle.src.utils.distances import distance_all_to_all


###############################################
### Isotropic gaussian kernel
class Gaussian(BaseKernel):
    def __init__(self,
                 spaces: EnvSpaces,
                 pms: Optional[SimpleNamespace]=None) -> None:
        super().__init__(spaces)

        self.dim_ = 1

    # Compute isotropic covariance function
    # xi and xj have shapes (n_batch, dim)
    def covariance(self,
                   xi: ndarray,
                   xj: ndarray,
                   theta: ndarray) -> ndarray:

        dist = distance_all_to_all(xi, xj)
        v    = np.exp(-0.5*(dist/theta[0])**2)

        return v
