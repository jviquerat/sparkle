# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.pex.pex    import pex_factory
from sparkle.src.env.spaces import environment_spaces

# Sample using pex
def sample(pex_type, n_points):

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    space = environment_spaces(loc_space)

    pex = pex_factory.create(pex_type,
                             spaces = space,
                             pms    = pms)

    pex.summary()
    pex.render_2d()
