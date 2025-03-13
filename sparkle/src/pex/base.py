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

    # Compute distance to nearest neighbour for each point
    def dist_nearest(self, x):

        nearest = np.zeros(self.n_points)
        for i in range(self.n_points):
            dist_min = 1.0e8
            for j in range(self.n_points):
                if (i==j): continue

                dist = np.linalg.norm(x[i] - x[j])
                if (dist < dist_min):
                    dist_min = dist

            nearest[i] = dist_min

        return nearest

    # Compute minimal distance between two points
    def dist_min(self, x):

        dmin = 1.0e8
        for i in range(self.n_points):
            for j in range(i+1, self.n_points):
                dist = np.linalg.norm(x[i] - x[j])
                if (dist < dmin): dmin = dist

        return dmin

    # Print informations
    def summary(self):

        spacer("Pex type is "+self.name_+" with "+str(self.n_points)+" points")

        nearest = self.dist_nearest(self.x)
        d_mean  = np.mean(nearest)
        d_std   = np.std(nearest)

        spacer("Mean nearest neighbor distance: "+fmt_float(d_mean))
        spacer("Std  nearest neighbor distance: "+fmt_float(d_std))

    # 2D rendering (for debugging purpose)
    def render_2d(self):

        if (self.dim != 2): return

        nearest = self.dist_nearest(self.x)
        nearest /= np.max(nearest)

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(5, 5)
        fig.subplots_adjust(0.01,0.01,0.99,0.95)

        ax.set_title(self.name_)
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.5)

        cmap = matplotlib.cm.RdBu
        ax.scatter(self.x[:,0], self.x[:,1], c=cmap(nearest), marker="o", alpha=0.8)
        plt.savefig(self.name_, dpi=100)
        plt.close()
