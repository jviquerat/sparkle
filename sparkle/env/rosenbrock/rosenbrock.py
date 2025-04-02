import numpy as np

from sparkle.env.base_env import base_env


###############################################
### Environment for 2D rosenbrock
class rosenbrock(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'rosenbrock'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2
        if hasattr(pms, "dim"): self.dim = pms.dim

        self.x0        =-1.0*np.ones(self.dim)
        self.x0[0]     = 0.0
        self.xmin      =-2.0*np.ones(self.dim)
        self.xmax      = 2.0*np.ones(self.dim)
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 500.0
        self.levels    = [1.0, 10.0, 50.0, 200.0, 500.0]

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        v = 0.0
        for i in range(self.dim-1):
            v += 100.0*(x[i+1]-x[i]**2)**2 + (1.0-x[i])**2

        return v

    # Close environment
    def close(self):
        pass
