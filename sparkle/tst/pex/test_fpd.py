# Generic imports
import os
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.pex.fpd    import fpd
from sparkle.src.env.spaces import env_spaces

###############################################
### Test fixed_poisson_disc pex
def test_fpd():

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    pms      = types.SimpleNamespace()

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = env_spaces(loc_space)

    # We test different number of points as fixed_poisson_disc may
    # have issues providing the exact number of required points
    pms.n_points = 2
    pex = fpd(s, pms)
    assert(pex.n_points == 2)

    pms.n_points = 10
    pex = fpd(s, pms)
    assert(pex.n_points == 10)

    pms.n_points = 100
    pex = fpd(s, pms)
    assert(pex.n_points == 100)

    pex.render_2d()
    filename = pex.name_+".png"
    os.remove(filename)
