# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.kernel.base import base_kernel

###############################################
### Matern 5/2 kernel
class matern52(base_kernel):
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
        v0   = (1.0 + math.sqrt(5.0)*dist/theta[0] + 5.0/(3.0*theta[0])*dist**2)
        val  = v0*np.exp(-math.sqrt(5.0)*dist/theta[0])

        return val
