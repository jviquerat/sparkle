# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.pex.pex    import pex_factory
from sparkle.src.env.spaces import env_spaces

# Sample using pex
def sample(pex_type, n_points, dim):

    xmin     = np.zeros(dim)
    xmax     = np.ones(dim)

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    space = env_spaces(loc_space)

    pex = pex_factory.create(pex_type,
                             spaces = space,
                             pms    = pms)

    pex.summary()

    if (dim == 2): pex.render_2d()
