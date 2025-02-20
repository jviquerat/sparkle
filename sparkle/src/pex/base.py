# Generic imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.utils.prints import spacer

###############################################
### Base experiment plan
class base_pex():
    def __init__(self, spaces):

        self.spaces = spaces

    @property
    def dim(self):
        return self.spaces.dim

    @property
    def natural_dim(self):
        return self.spaces.natural_dim

    @property
    def x0(self):
        return self.spaces.x0

    @property
    def xmin(self):
        return self.spaces.xmin

    @property
    def xmax(self):
        return self.spaces.xmax

    # Return total nb of points
    def n_points(self):

        return self.x().shape[0]

    # Return i-th point of pex
    def point(self, i):

        return np.array([self.x_[i]])

    # Return pex points
    def x(self):

        return self.x_

    # Compute volume of domain
    def volume(self):

        v = self.xmax - self.xmin
        return np.prod(v)

    # Print informations
    def summary(self):

        spacer()
        print("Pex type is "+self.name_+" with "+str(self.n_points())+" points")

    # 2D rendering (for debugging purpose)
    def render_2d(self):

        if (self.dim != 2): return

        plt.clf()
        fig = plt.figure()
        plt.xlim([self.xmin[0], self.xmax[0]])
        plt.ylim([self.xmin[1], self.xmax[1]])
        plt.grid()
        plt.scatter(self.x_[:,0], self.x_[:,1], c="black", marker="o")
        plt.savefig(self.name_, dpi=100)
        plt.close()
