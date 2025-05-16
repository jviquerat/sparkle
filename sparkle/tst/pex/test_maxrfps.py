import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.maxrfps import MaxRFPS
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_MaxRFPS():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)
    pex = MaxRFPS(s, pms)

    ref = np.array([[0.4236548 , 0.64589411],
                    [0.98837384, 0.10204481],
                    [0.07103606, 0.0871293 ],
                    [0.97861834, 0.79915856],
                    [0.52324805, 0.09394051],
                    [0.0202184 , 0.83261985],
                    [0.97645947, 0.4686512 ],
                    [0.5759465 , 0.9292962 ],
                    [0.19658236, 0.36872517],
                    [0.57019677, 0.43860151]])

    assert pex.n_points == n_points
    assert np.allclose(ref, pex.x)
