import math

import numpy as np

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default


###############################################
### Environment for griewank
class griewank(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'griewank'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = set_default("dim", 2, pms)
        self.x0        = set_default("x0", 5.0*np.ones(self.dim), pms)
        self.xmin      = set_default("xmin", -10.0*np.ones(self.dim), pms)
        self.xmax      = set_default("xmax", 10.0*np.ones(self.dim), pms)

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

        s = 0.0
        p = 1.0
        for i in range(self.dim):
            s += x[i]**2
            p *= math.cos(x[i]/math.sqrt(1.0+i))

        return 1.0 + s/4000.0 - p

    # Close environment
    def close(self):
        pass
