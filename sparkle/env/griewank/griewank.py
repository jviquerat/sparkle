# Generic imports
import numpy as np

# Custom imports
from sparkle.env.base_env import base_env

###############################################
### Environment for griewank
class griewank(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'griewank'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2
        self.x0        =  5.0*np.ones(self.dim)
        self.xmin      =-10.0*np.ones(self.dim)
        self.xmax      = 10.0*np.ones(self.dim)
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 2.0
        self.levels    = [1.0]

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        return 1.0 + (x[0]**2+x[1]**2)/4000.0 - math.cos(x[0])*math.cos(x[1]/math.sqrt(2.0))

    # Close environment
    def close(self):
        pass
