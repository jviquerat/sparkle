# Generic imports
import os
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.pex.fpd     import fpd
from sparkle.src.env.spaces  import env_spaces
from sparkle.src.utils.seeds import set_seeds

###############################################
### Test fixed_poisson_disc pex
def test_fpd():

    set_seeds(0)

    dim  = 2
    xmin = np.array([0.0, 0.0])
    xmax = np.array([1.0, 1.0])
    pms  = types.SimpleNamespace()

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = env_spaces(loc_space)

    # We test different number of points as fixed_poisson_disc may
    # have issues providing the exact number of required points
    pms.n_points = 2
    pex = fpd(s, pms)
    assert(pex.n_points == 2)

    pms.n_points = 10
    pex = fpd(s, pms)

    ref = np.array([[0.56886585, 0.01429965],
                    [0.25699698, 0.98596387],
                    [0.99290718, 0.60222846],
                    [0.09396511, 0.35028631],
                    [0.50906054, 0.53534002],
                    [0.96048255, 0.06721008],
                    [0.83415342, 0.95817021],
                    [0.20595891, 0.69828514],
                    [0.02362962, 0.08950589],
                    [0.50839039, 0.96725931]])

    assert(pex.n_points == 10)
    assert(np.allclose(ref, pex.x))

    pms.n_points = 100
    pex = fpd(s, pms)
    assert(pex.n_points == 100)
