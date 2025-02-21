# Generic imports
import os
import pytest
import types
import numpy as np

# Custom imports
from sparkle.tst.tst                    import *
from sparkle.src.pex.fixed_poisson_disc import fixed_poisson_disc
from sparkle.src.env.spaces             import environment_spaces

###############################################
### Test fixed_poisson_disc pex
def test_fixed_poisson_disc():

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    pms      = types.SimpleNamespace()

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = environment_spaces(loc_space)

    # We test different number of points as fixed_poisson_disc may
    # have issues providing the exact number of required points
    pms.n_points = 2
    pex = fixed_poisson_disc(s, pms)
    assert(pex.n_points == 2)

    pms.n_points = 10
    pex = fixed_poisson_disc(s, pms)
    assert(pex.n_points == 10)

    pms.n_points = 100
    pex = fixed_poisson_disc(s, pms)
    assert(pex.n_points == 100)

    pex.render_2d()
    filename = pex.name_+".png"
    os.remove(filename)
