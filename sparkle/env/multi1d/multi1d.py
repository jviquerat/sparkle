# Generic imports
from  math import sin

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
        nx = 200
        self.x = np.linspace(self.xmin[0], self.xmax[0], num=nx)
        self.z = np.zeros((nx))
        for i in range(nx):
            self.z[i] = self.cost([self.x[i]])

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
    def render(self, x):

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

        plt.clf()
        fig, ax = plt.subplots()
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([-2.0, 2.5])
        fig.subplots_adjust(0,0,1,1)
        plt.plot(self.x, self.z)

        z = np.zeros_like(x)
        for i in range(len(x)):
            z[i] = self.cost(x[i])

        plt.scatter(x[:,0], z[:,0], c="black", marker='o', alpha=0.8)

        filename = self.path+"/png/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

    # Close environment
    def close(self):
        pass
