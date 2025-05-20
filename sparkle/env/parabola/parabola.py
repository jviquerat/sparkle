import numpy as np

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default


###############################################
class parabola(base_env):
    """
    Standard parabola function in arbitrary dimension
    """
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'parabola'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = set_default("dim", 2, pms)

        self.x0        = set_default("x0",   2.5*np.ones(self.dim), pms)
        self.xmin      = set_default("xmin",-5.0*np.ones(self.dim), pms)
        self.xmax      = set_default("xmax", 5.0*np.ones(self.dim), pms)

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

        v = 0.0
        for i in range(self.dim):
            v += (x[i])**2

        return v

    def close(self):
        pass
