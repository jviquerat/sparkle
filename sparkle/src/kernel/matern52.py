import math
import numpy as np
from numpy import ndarray
from types import SimpleNamespace
from typing import Optional

from sparkle.src.kernel.base     import BaseKernel
from sparkle.src.utils.distances import distance_all_to_all
from sparkle.src.env.spaces import EnvSpaces

###############################################
### Isotropic Matern 5/2 kernel
class Matern52(BaseKernel):
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
        v0   = (1.0 + math.sqrt(5.0)*dist/theta[0] + 5.0/(3.0*theta[0])*dist**2)
        v    = v0*np.exp(-math.sqrt(5.0)*dist/theta[0])

        return v
