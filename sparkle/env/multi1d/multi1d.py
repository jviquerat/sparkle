# Generic imports
from math import sin

# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for 1D multimodal function
class multi1d(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'multi1d'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 1
        self.x0        = np.array([0.6])
        self.xmin      = np.array([0.0])
        self.xmax      = np.array([1.2])
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Plotting data
        self.it_plt    = 0
        self.vmin      =-2.0
        self.vmax      = 3.0

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        v = (3.0*x[0] - 1.4)*sin(18.0*x[0])

        return v

    # Close environment
    def close(self):
        pass
