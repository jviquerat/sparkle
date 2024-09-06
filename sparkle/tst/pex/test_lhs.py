# Generic imports
import pytest
import types
import numpy as np

# Custom imports
from sparkle.tst.tst     import *
from sparkle.src.pex.lhs import *

###############################################
### Test lhs pex
def test_lhs():

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 12

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    pex = lhs(dim, xmin, xmax, pms)
    assert(pex.x_.shape[0] == n_points)
    #pex.render_2d()
