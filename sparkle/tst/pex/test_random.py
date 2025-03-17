# Generic imports
import os
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.pex.random import random
from sparkle.src.env.spaces import env_spaces

###############################################
### Test random pex
def test_random():

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = env_spaces(loc_space)
    pex = random(s, pms)
    assert(pex.n_points == n_points)
    pex.render_2d()
    filename = pex.name+".png"
    os.remove(filename)
