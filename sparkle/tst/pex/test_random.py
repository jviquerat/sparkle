# Generic imports
import os
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.pex.random  import random
from sparkle.src.env.spaces  import env_spaces
from sparkle.src.utils.seeds import set_seeds

###############################################
### Test random pex
def test_random():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = env_spaces(loc_space)
    pex = random(s, pms)

    ref = np.array([[0.5488135,  0.71518937],
                    [0.60276338, 0.54488318],
                    [0.4236548,  0.64589411],
                    [0.43758721, 0.891773  ],
                    [0.96366276, 0.38344152],
                    [0.79172504, 0.52889492],
                    [0.56804456, 0.92559664],
                    [0.07103606, 0.0871293 ],
                    [0.0202184,  0.83261985],
                    [0.77815675, 0.87001215]])

    assert pex.n_points == n_points
    assert np.allclose(ref, pex.x)
