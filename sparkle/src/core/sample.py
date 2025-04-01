# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.pex.pex     import pex_factory
from sparkle.src.env.spaces  import EnvSpaces
from sparkle.src.utils.timer import Timer

# Sample using pex
def sample(pex_type, n_points, dim):

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

    if (dim == 2): pex.render_2d()
