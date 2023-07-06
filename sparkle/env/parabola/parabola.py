# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.env.base_env import base_env

###############################################
### Environment for parabola
class parabola(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name  = 'parabola'
        self.path  = path
        self.cpu   = cpu
        self.dim   = 2
        self.xmin  = np.array([-5.0,-5.0])
        self.xmax  = np.array([ 5.0, 5.0])
        self.x0    = np.array([ 2.5, 2.5])

        # Check inputs
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax
        if hasattr(pms, "x0"):   self.xmin = pms.x0

    # Reset environment
    def reset(self):

        return True

    # Cost function
    def cost(self, x):

        # Scale inputs
        sx = self.scale(x)

        # Compute function value in x
        v = 0.0
        for i in range(len(x)):
            v += (sx[i])**2

        return v

    # Scale parameters
    def scale(self, x):

        # Scale
        sx = self.dim*[None]
        xp = self.xmax - self.x0
        xm = self.x0   - self.xmin

        for i in range(self.dim):
            if (x[i] >= 0.0):
                sx[i] = self.x0[i] + xp[i]*x[i]
            if (x[i] <  0.0):
                sx[i] = self.x0[i] + xm[i]*x[i]

        return sx

    # Rendering
    def render(self):

        return True

    # Close environment
    def close(self):
        pass
