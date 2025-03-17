# Generic imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.utils.prints import spacer, fmt_float

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

    @property
    def x(self):
        return self.x_

    @property
    def n_points(self):
        return self.x.shape[0]

    # Return i-th point of pex
    def point(self, i):

        return np.array([self.x[i]])

    # Compute volume of domain
    def volume(self):

        v = self.xmax - self.xmin
        return np.prod(v)

    # Compute nearest neighbour for all input coordinates
    def nearest(self, x):

        n_points  = x.shape[0]
        d_nearest = np.zeros(n_points)
        p_nearest = np.zeros(n_points, dtype=int)

        for i in range(n_points):
            d_min, p_min = self.p_nearest(x, i)
            d_nearest[i] = d_min
            p_nearest[i] = p_min

        return d_nearest, p_nearest

    # Compute nearest neighbour for one input coordinates
    def p_nearest(self, x, i):

        n_points = x.shape[0]
        d_min    = 1.0e8
        p_min    =-1

        for j in range(n_points):
            if (i==j): continue

            d = self.distance(x[i], x[j])
            if (d < d_min):
                d_min = d
                p_min = j

        return d_min, p_min

    # Compute minimal distance between two points
    def min_distance(self, x):

        n_points = x.shape[0]
        dmin     = 1.0e8

        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = self.distance(x[i], x[j])
                if (dist < dmin): dmin = dist

        return dmin

    # Compute distance between two sets of coordinates
    def distance(self, xi, xj):

        return np.linalg.norm(xi - xj)

    # Print informations
    def summary(self):

        spacer("Pex type is "+self.name+" with "+str(self.n_points)+" points")

        d_nearest, _ = self.nearest(self.x)
        d_mean       = np.mean(d_nearest)
        d_std        = np.std(d_nearest)

        spacer("Mean nearest neighbor distance: "+fmt_float(d_mean))
        spacer("Std  nearest neighbor distance: "+fmt_float(d_std))

    # 2D rendering (for debugging purpose)
    def render_2d(self):

        if (self.dim != 2): return

        d_nearest, _ = self.nearest(self.x)
        d_nearest /= np.max(d_nearest)

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(5, 5)
        fig.subplots_adjust(0.01,0.01,0.99,0.95)

        ax.set_title(self.name)
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.5)

        cmap = matplotlib.cm.RdBu
        ax.scatter(self.x[:,0], self.x[:,1], c=cmap(d_nearest), marker="o", alpha=0.8)
        plt.savefig(self.name, dpi=100)
        plt.close()
