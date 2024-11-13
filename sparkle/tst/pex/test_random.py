# Generic imports
import os
import pytest
import types
import numpy as np

# Custom imports
from sparkle.tst.tst        import *
from sparkle.src.pex.random import *

###############################################
### Test random pex
def test_random():

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    pex = random(dim, xmin, xmax, pms)
    assert(pex.n_points() == n_points)
    pex.render_2d()
    os.remove(pex.render_2d_filename)
