import math
import numpy as np

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default


###############################################
class mixed(base_env):
    """
    A mixed continuous/discrete problem
    """
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'mixed'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = set_default("dim", 3, pms)

        self.continuous_dim = 2
        self.discrete_dim = 1
        self.discrete_cats = [[-3, -2, -1, 0, 1, 2, 3]]

        self.x0        = set_default("x0",   0.0*np.ones(self.continuous_dim), pms)
        self.xmin      = set_default("xmin",-2.0*np.ones(self.continuous_dim), pms)
        self.xmax      = set_default("xmax", 2.0*np.ones(self.continuous_dim), pms)

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 20.0
        self.levels    = [0.1, 1.0, 5.0, 10.0, 20.0]

    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    def cost(self, x):

        xc0 = x[0]
        xc1 = x[1]
        xd  = self.discrete_cats[0][int(x[2])]

        v = (xc0 - 2.0*xd)**2 + (xc1 - 0.5*xd)**2 + 8.0*math.cos(math.pi*xd) + xd**2 + 2.0*xd

        return v

    def close(self):
        pass
