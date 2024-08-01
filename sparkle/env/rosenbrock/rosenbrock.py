# Custom imports
from sparkle.env.base_env import *

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
        self.it_plt    = 0

        # Check inputs
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Generate map of cost values for rendering
        if (self.dim != 2): return
        self.generate_cost_map_2d()

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        v = 0.0
        for i in range(len(x)-1):
            v += 100.0*(x[i+1]-x[i]**2)**2 + (1.0-x[i])**2

        return v

    # Rendering
    def render(self, x, y_mu=None, y_std=None, x_ei=None):

        if (self.dim != 2): return

        self.render_2d(x, vmin=0, vmax=200,
                       levels=[1.0, 10.0, 50.0, 200.0, 500.0],
                       y_mu=y_mu, y_std=y_std, x_ei=x_ei)

    # Close environment
    def close(self):
        pass
