# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.kernel.base     import base_kernel
from sparkle.src.utils.distances import distance_all_to_all

###############################################
### Isotropic Matern 5/2 kernel
class matern52(base_kernel):
    def __init__(self, spaces, pms=None):
        super().__init__(spaces)

        self.dim_ = 1

    # Compute isotropic covariance function
    # xi and xj have shapes (n_batch, dim)
    def covariance(self, xi, xj, theta):

        dist = distance_all_to_all(xi, xj)
        v0   = (1.0 + math.sqrt(5.0)*dist/theta[0] + 5.0/(3.0*theta[0])*dist**2)
        v    = v0*np.exp(-math.sqrt(5.0)*dist/theta[0])

        return v
