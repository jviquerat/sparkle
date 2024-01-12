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

        nx = 100
        ny = 100

        self.x         = np.linspace(self.xmin[0], self.xmax[0], num=100)
        self.y         = np.linspace(self.xmax[1], self.xmin[1], num=100)
        self.x, self.y = np.array(np.meshgrid(self.x, self.y))
        self.z         = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                self.z[i,j] = self.cost([self.x[i,j], self.y[i,j]])

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
    def render(self, x):

        if (self.dim != 2): return

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(self.z))
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(self.z,
                   extent=[self.xmin[0], self.xmax[0],
                           self.xmin[1], self.xmax[1]],
                   vmin=0.0, vmax=200.0,
                   alpha=0.8, cmap='RdBu_r')
        cnt = plt.contour(self.x, self.y, self.z,
                          levels=[1.0, 10.0, 50.0, 200.0, 500.0],
                          colors='black', alpha=0.5)
        plt.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
        plt.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)

        filename = self.path+"/png/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

    # Close environment
    def close(self):
        pass
