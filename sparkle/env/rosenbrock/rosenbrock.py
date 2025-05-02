import numpy as np

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default


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
        self.dim       = set_default("dim", 2, pms)

        x0        =-1.0*np.ones(self.dim)
        x0[0]     = 0.0
        self.x0   = set_default("x0", x0, pms)
        self.xmin = set_default("xmin", -2.0*np.ones(self.dim), pms)
        self.xmax = set_default("xmax", 2.0*np.ones(self.dim), pms)

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
