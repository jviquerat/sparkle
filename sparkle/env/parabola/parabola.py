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
        self.x0     = np.array([ 2.5, 2.5])
        self.it_plt = 0

        # Check inputs
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax
        if hasattr(pms, "x0"):   self.xmin = pms.x0

        # Generate map of cost values for rendering
        nx = 100
        ny = 100

        self.x         = np.linspace(self.xmin[0], self.xmax[0], 100)
        self.y         = np.linspace(self.xmin[1], self.xmax[1], 100)
        self.x, self.y = np.array(np.meshgrid(self.x, self.y))
        self.z         = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                self.z[i,j] = self.f([self.x[i,j], self.y[i,j]])

    # Reset environment
    def reset(self):

        return True

    # Cost function
    def cost(self, x):

        # Scale inputs
        sx = self.scale(x)

        # Compute value for input x
        return self.f(x)

    # Function
    def f(self, x):

        v = 0.0
        for i in range(len(x)):
            v += (x[i])**2

        return v

    # Scale parameters
    def scale(self, x):

        # Scale
        sx = self.dim*[None]
        xp = self.xmax - self.x0
        xm = self.x0   - self.xmin

        for i in range(self.dim):
            if (x[i] >= 0.0):
                sx[i] = self.x0[i] + xp[i]*x[i]
            if (x[i] <  0.0):
                sx[i] = self.x0[i] + xm[i]*x[i]

        return sx

    # Rendering
    def render(self, x):

        print("coucou")
        # Set up base figure: The contour map
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(self.z))

        fig.subplots_adjust(0,0,1,1)
        plt.imshow(self.z,
                   cmap = 'RdBu_r')

        plt.scatter(x[:,0], x[:,1], marker='o')

        print(self.path)
        filename = self.path+"/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

        # fig, ax = plt.subplots(figsize=(5,5))
        # fig.set_tight_layout(True)
        # img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
        # fig.colorbar(img, ax=ax)
        # ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
        # contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
        # ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        # pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
        # p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
        # p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
        # gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
        # ax.set_xlim([0,5])
        # ax.set_ylim([0,5])

    # Close environment
    def close(self):
        pass
