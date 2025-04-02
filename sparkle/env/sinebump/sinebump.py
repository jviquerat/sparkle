import numpy as np

from sparkle.env.base_env import base_env

###############################################
### Environment for 2D sinebump
class sinebump(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'sinebump'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2
        self.x0        = np.array([2.5, 2.5])
        self.xmin      = np.array([0.0, 0.0])
        self.xmax      = np.array([5.0, 5.0])
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 16.0
        self.levels    = [0.0, 2.0, 4.0, 6.0, 8.0]

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        v = (x[0]-3.14)**2 + (x[1]-2.72)**2 + np.sin(3*x[0]+1.41) + np.sin(4*x[1]-1.73)

        return v

    # Close environment
    def close(self):
        pass
