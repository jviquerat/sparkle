# Generic imports
import numpy as np

# Custom imports
from sparkle.src.kernel.base     import BaseKernel
from sparkle.src.utils.distances import distance_all_to_all

###############################################
### Isotropic gaussian kernel
class Gaussian(BaseKernel):
    def __init__(self, spaces, pms=None):
        super().__init__(spaces)

        self.dim_ = 1

    # Compute isotropic covariance function
    # xi and xj have shapes (n_batch, dim)
    def covariance(self, xi, xj, theta):

        dist = distance_all_to_all(xi, xj)
        v    = np.exp(-0.5*(dist/theta[0])**2)

        return v
