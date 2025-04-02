import numpy as np

from sparkle.env.base_env import base_env

###############################################
### Environment for branin
class branin(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'branin'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2
        self.x0        = np.array([ 7.5,  7.5])
        self.xmin      = np.array([ 0.0,  0.0])
        self.xmax      = np.array([15.0, 15.0])
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 100.0
        self.levels    = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        a = 1.0
        b = 5.1/(4.0*math.pi**2)
        c = 5.0/math.pi
        r = 6.0
        s = 10.0
        t = 1.0/(8.0*math.pi)

        return a*(x[1]-b*x[0]**2+c*x[0]-r)**2 + s*(1.0-t)*math.cos(x[0]) + s

    # Close environment
    def close(self):
        pass
