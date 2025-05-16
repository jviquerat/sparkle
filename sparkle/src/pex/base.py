import math
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import float64, ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default
from sparkle.src.utils.distances import nearest_neighbors_in_set, pairwise_distances
from sparkle.src.utils.prints import fmt_float, spacer


###############################################
class BasePex():
    """
    Base class for experiment plans.

    This class defines the common interface and functionality for all
    experiment plans (Pex) used in the optimization framework. It provides
    methods for managing the search space, generating points, computing
    quality metrics, and rendering the experiment plan.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the BasePex.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan,
                including the number of points (n_points).
        """

        self.spaces    = spaces
        self.n_points_ = set_default("n_points", 10*self.dim, pms)
        self.MC_set    = None

    @property
    def dim(self) -> int:
        """
        Returns the dimensionality of the search space.
        """
        return self.spaces.dim

    @property
    def natural_dim(self) -> int:
        """
        Returns the natural dimensionality of the search space.
        """
        return self.spaces.natural_dim

    @property
    def x0(self) -> ndarray:
        """
        Returns the initial point in the search space.
        """
        return self.spaces.x0

    @property
    def xmin(self) -> ndarray:
        """
        Returns the lower bounds of the search space.
        """
        return self.spaces.xmin

    @property
    def xmax(self) -> ndarray:
        """
        Returns the upper bounds of the search space.
        """
        return self.spaces.xmax

    @property
    def x(self) -> ndarray:
        """
        Returns the generated points of the experiment plan.
        """
        return self.x_

    @property
    def n_points(self) -> int:
        """
        Returns the number of points in the experiment plan.
        """
        return self.x.shape[0]

    def point(self, i: int) -> ndarray:
        """
        Returns the i-th point of the experiment plan.

        Args:
            i: The index of the point to retrieve.

        Returns:
            A NumPy array representing the i-th point.
        """

        return np.array([self.x[i]])

    def volume(self) -> float64:
        """
        Computes the volume of the search space domain.

        Returns:
            The volume of the search space.
        """

        v = self.xmax - self.xmin
        return np.prod(v)

    def phi_p(self, p: int=50) -> float:
        """
        Computes the phi-p criterion for the experiment plan.

        The phi-p criterion is a measure of the space-filling properties
        of the experiment plan. Default value suggested by Morris & Mitchell (1995)

        Args:
            p: The power parameter for the phi-p criterion.

        Returns:
            The phi-p criterion value.
        """

        d = 0.0
        for i in range(self.n_points):
            dists = np.linalg.norm(self.x[i+1:] - self.x[i], axis=1)
            d    += np.sum(np.power(dists, -p))

        return math.pow(d, 1.0/p)

    def minimax(self, x=None, n_samples: int=10000) -> float:
        """
        Computes the minimax criterion for the experiment plan,
        using Monte-Carlo sampling. We uniformly draw n_samples
        within the domain, and for each random point, find the minimum
        Euclidian distance to any of the selected design points

        Args:
            n_samples: the number of samples to draw for MC estimate

        Returns:
            The evaluated minimax distance
        """
        # Generate n_samples random points uniformly within the domain
        if (self.MC_set is None):
            self.MC_set = np.random.uniform(low=self.xmin,
                                            high=self.xmax,
                                            size=(n_samples, self.dim))

        if x is None: x = self.x

        # For each Monte Carlo point, find its minimum squared Euclidean distance
        # to any of the selected design points (self.x)
        dists        = pairwise_distances(self.MC_set, x)
        minimax_dist = np.max(np.min(dists, axis=1))

        return minimax_dist

    def summary(self):
        """
        Prints a summary of the experiment plan's configuration.
        """

        spacer("Pex type is "+self.name+" with "+str(self.n_points)+" points")
        spacer("Phi-p criterion: "+fmt_float(self.phi_p()))
        spacer("Minimax criterion: "+fmt_float(self.minimax()))

    def render_distances_distributions(self, x=None, i: int=None):
        """
        Compute distance distributions
        """
        if x is None: x = self.x

        dists = pairwise_distances(x, x)
        dists = np.reshape(dists, (-1))

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(5, 5)

        filename = f"{self.name}_hist_pairwise.png"
        if i is not None: filename = f"{self.name}_hist_pairwise_{i}.png"
        plt.hist(dists)
        plt.savefig(filename, dpi=100)
        plt.close()

        dists, _ = nearest_neighbors_in_set(x)

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(5, 5)

        filename = f"{self.name}_hist_neighbors.png"
        if i is not None: filename = f"{self.name}_hist_neighbors_{i}.png"
        plt.hist(dists)
        plt.savefig(filename, dpi=100)
        plt.close()

    def render_2d(self):
        """
        Renders the experiment plan in 2D (if the dimensionality is 2).

        This method generates a 2D scatter plot of the points in the
        experiment plan, colored by their nearest neighbor distances.
        It is primarily used for debugging and visualization purposes.
        """

        if (self.dim != 2): return

        d_nearest, _ = nearest_neighbors_in_set(self.x)
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
