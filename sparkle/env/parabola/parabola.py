# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for parabola
class parabola(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'parabola'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2
        if hasattr(pms, "dim"): self.dim = pms.dim

        self.x0        = 2.5*np.ones(self.dim)
        self.xmin      =-5.0*np.ones(self.dim)
        self.xmax      = 5.0*np.ones(self.dim)
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 20.0
        self.levels    = [0.1, 1.0, 5.0, 10.0, 20.0]

        # Generate map of cost values for rendering
        if (self.dim == 1): self.generate_cost_map_1d()
        if (self.dim == 2): self.generate_cost_map_2d()

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        v = 0.0
        for i in range(len(x)):
            v += (x[i])**2

        return v

    # Rendering
    def render(self, x, c, pms=None):

        if (self.dim == 1): self.render_1d(x, pms=pms)
        if (self.dim == 2): self.render_2d(x, pms=pms)
        if (self.dim >  2): return

    # Close environment
    def close(self):
        pass
