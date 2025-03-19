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

    ref = np.array([[0.3156184 , 0.38976716],
                    [0.72736339, 0.99481176],
                    [0.90725543, 0.19721412],
                    [0.06756243, 0.74752168],
                    [0.31356976, 0.98511461],
                    [0.91350519, 0.50544328],
                    [0.66697423, 0.36335019],
                    [0.06514091, 0.26988353],
                    [0.50596902, 0.79100612],
                    [0.69748725, 0.04744814]])

    assert(pex.n_points == 10)
    assert(np.allclose(ref, pex.x))

    pms.n_points = 100
    pex = fpd(s, pms)
    assert(pex.n_points == 100)
