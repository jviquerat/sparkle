# Generic imports
from math import sin

# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for 1D multimodal function
class multi1d(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'multi1d'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 1
        self.x0        = np.array([0.6])
        self.xmin      = np.array([0.0])
        self.xmax      = np.array([1.2])
        self.it_plt    = 0

        # Check inputs
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Generate map of cost values for rendering
        self.nx_plot = 200
        self.x_plot  = np.linspace(self.xmin[0], self.xmax[0], num=self.nx_plot)
        self.y_plot  = np.zeros(self.nx_plot)
        for i in range(self.nx_plot):
            self.y_plot[i] = self.cost([self.x_plot[i]])

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        v = (3.0*x[0] - 1.4)*sin(18.0*x[0])

        return v

    # Rendering
    def render(self, x, x_mu=None, y_mu=None, y_std=None, ei=None, x_ei=None):

        self.render_1d(x, x_mu=x_mu, y_mu=y_mu, y_std=y_std, ei=ei, x_ei=x_ei)

    # Close environment
    def close(self):
        pass
