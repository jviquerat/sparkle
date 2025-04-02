import math
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import float64, ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default
from sparkle.src.utils.distances import nearest_all_to_all
from sparkle.src.utils.prints import fmt_float, spacer


###############################################
### Base experiment plan
class BasePex():
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:

        self.spaces    = spaces
        self.n_points_ = set_default("n_points", 10*self.dim, pms)

    @property
    def dim(self) -> int:
        return self.spaces.dim

    @property
    def natural_dim(self) -> int:
        return self.spaces.natural_dim

    @property
    def x0(self) -> ndarray:
        return self.spaces.x0

    @property
    def xmin(self) -> ndarray:
        return self.spaces.xmin

    @property
    def xmax(self) -> ndarray:
        return self.spaces.xmax

    @property
    def x(self) -> ndarray:
        return self.x_

    @property
    def n_points(self) -> int:
        return self.x.shape[0]

    # Return i-th point of pex
    def point(self, i: int) -> ndarray:

        return np.array([self.x[i]])

    # Compute volume of domain
    def volume(self) -> float64:

        v = self.xmax - self.xmin
        return np.prod(v)

    # Compute phi-p criterion
    # Default value suggested by Morris & Mitchell (1995)
    def phi_p(self, p: int=50) -> float:

        d = 0.0
        for i in range(self.n_points):
            dists = np.linalg.norm(self.x[i+1:] - self.x[i], axis=1)
            d    += np.sum(np.power(dists, -p))

        return math.pow(d, 1.0/p)

    # Print informations
    def summary(self):

        spacer("Pex type is "+self.name+" with "+str(self.n_points)+" points")
        spacer("Phi-p criterion: "+fmt_float(self.phi_p()))

    # 2D rendering (for debugging purpose)
    def render_2d(self):

        if (self.dim != 2): return

        d_nearest, _ = nearest_all_to_all(self.x)
        d_nearest   /= np.max(d_nearest)

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
