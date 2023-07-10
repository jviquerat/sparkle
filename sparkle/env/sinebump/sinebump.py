# Generic imports
from   math  import cos, sqrt
import numpy as     np

# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for 2D sinebump
class sinebump(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name   = 'sinebump'
        self.path   = path
        self.cpu    = cpu
        self.dim    = 2
        self.xmin   = np.array([0.0, 0.0])
        self.xmax   = np.array([5.0, 5.0])
        self.it_plt = 0

        # Check inputs
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Generate map of cost values for rendering
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
    def reset(self):

        return True

    # Cost function
    def cost(self, x):

        v = (x[0]-3.14)**2 + (x[1]-2.72)**2 + np.sin(3*x[0]+1.41) + np.sin(4*x[1]-1.73)

        return v

    # Rendering
    def render(self, x):

        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(self.z))
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(self.z,
                   extent=[self.xmin[0], self.xmax[0],
                           self.xmin[1], self.xmax[1]],
                   vmin=0.0, vmax=16.0,
                   alpha=0.8, cmap='RdBu_r')
        cnt = plt.contour(self.x, self.y, self.z,
                          levels=[0, 2, 4, 6, 8],
                          colors='black', alpha=0.5)
        plt.clabel(cnt, inline=True, fontsize=8, fmt="%.1f")
        plt.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)

        filename = self.path+"/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

    # Close environment
    def close(self):
        pass
