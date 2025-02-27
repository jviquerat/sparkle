# Generic imports
import numpy as np

# Custom imports
from sparkle.src.kernel.base   import base_kernel
from sparkle.src.utils.default import set_default

###############################################
### Gaussian kernel
class gaussian(base_kernel):
    def __init__(self, spaces, pms):
        super().__init__(spaces)

        self.dim_      = 1
        self.x0_       = np.zeros(self.dim_)
        self.xmin_     =-2.0*np.ones(self.dim_)
        self.xmax_     = np.zeros(self.dim_)
        self.diag_eps_ = 1.0e-15

        self.reset()

    def reset(self):

        self.theta_ = np.exp(0.5*(self.xmin_ + self.xmax_))

    # Compute isotropic covariance function
    def covariance(self, xi, xj, theta):

        dist = np.linalg.norm(xi-xj)
        v    = np.exp(-0.5*(dist/theta[0])**2)

        return v
