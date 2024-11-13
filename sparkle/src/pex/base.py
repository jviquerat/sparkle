# Generic imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.utils.prints import spacer

###############################################
### Base experiment plan
class base_pex():
    def __init__(self):

        self.render_2d_filename = "pex_render_2d.png"

    # Return total nb of points
    def n_points(self):

        return self.x().shape[0]

    # Return i-th point of pex
    def point(self, i):

        return np.array([self.x_[i]])

    # Return pex points
    def x(self):

        return self.x_

    # Print informations
    def summary(self):

        spacer()
        print("Pex type is "+self.name_+" with "+str(self.n_points())+" points")

    # 2D rendering (for debugging purpose)
    def render_2d(self):

        if (self.dim_ != 2): return

        plt.clf()
        fig = plt.figure()
        plt.xlim([self.xmin_[0], self.xmax_[0]])
        plt.ylim([self.xmin_[1], self.xmax_[1]])
        major_ticks = np.arange(self.xmin_[0], self.xmax_[0]+1.0e-8, 1.0/self.n_points_)
        plt.xticks(major_ticks)
        plt.yticks(major_ticks)
        plt.grid()
        plt.scatter(self.x_[:,0], self.x_[:,1], c="black", marker="o")
        plt.savefig(self.render_2d_filename, dpi=100)
        plt.close()
