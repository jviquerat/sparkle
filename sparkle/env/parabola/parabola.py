# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for parabola
class parabola(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name   = 'parabola'
        self.path   = path
        self.cpu    = cpu
        self.dim    = 2
        self.xmin   = np.array([-5.0,-5.0])
        self.xmax   = np.array([ 5.0, 5.0])
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

        v = 0.0
        for i in range(len(x)):
            v += (x[i])**2

        return v

    # Rendering
    def render(self, x):

        # Set up base figure: The contour map
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(self.z))
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(self.z,
                   extent=[self.xmin[0], self.xmax[0],
                           self.xmin[1], self.xmax[1]],
                   alpha=0.8, cmap='RdBu_r')
        cnt = plt.contour(self.x, self.y, self.z, 10,
                          colors='black', alpha=0.5)
        plt.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
        plt.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)

        filename = self.path+"/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

    # Close environment
    def close(self):
        pass
