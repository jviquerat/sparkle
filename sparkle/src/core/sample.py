import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.pex import pex_factory
from sparkle.src.utils.timer import Timer


def sample(pex_type, n_points, dim):
    """
    Samples points using a Point Exploration (Pex) algorithm.

    This function initializes a Pex algorithm, samples points from its
    distribution, and displays timing and summary information.

    Args:
        pex_type: The type of Pex algorithm to use (e.g., "random", "mlhs").
        n_points: The number of points to sample.
        dim: The dimensionality of the search space.
    """

    xmin     = np.zeros(dim)
    xmax     = np.ones(dim)

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    space = EnvSpaces(loc_space)

    pex_timer = Timer("pex_timer")
    pex_timer.tic()

    pex = pex_factory.create(pex_type,
                             spaces = space,
                             pms    = pms)
    pex_timer.toc()

    pex.summary()
    pex_timer.show()

    pex.render_distances_distributions()
    if (dim == 2): pex.render_2d()
