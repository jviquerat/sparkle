# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.env.base_env import base_env

###############################################
### Environment for parabola
class parabola(base_env):

    ### Create object
    def __init__(self, path, pms):

        # Fill structure
        self.name     = 'parabola'
        self.path     = path
        self.dim      = 2
        self.x_min    = np.array([-5.0,-5.0])
        self.x_max    = np.array([ 5.0, 5.0])
        self.x_0      = np.array([ 2.5, 2.5])

        # Check inputs
        if hasattr(pms, "x_min"): self.x_min = pms.x_min
        if hasattr(pms, "x_max"): self.x_max = pms.x_max
        if hasattr(pms, "x_0"):   self.x_min = pms.x_0

    ### Cost function
    def cost(self, x):

        # Compute function value in x
        v = 0.0
        for i in range(len(x)):
            v += (x[i])**2

        return v
    
    ### Rendering
    def render(self):
        pass
