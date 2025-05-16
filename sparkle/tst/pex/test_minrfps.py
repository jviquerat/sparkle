import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.minrfps import MinRFPS
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_MinRFPS():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)
    pex = MinRFPS(s, pms)

    ref = np.array([[0.61801543, 0.4287687 ],
                    [0.03330463, 0.95898272],
                    [0.02467873, 0.06724963],
                    [0.97291949, 0.96083466],
                    [0.98837384, 0.10204481],
                    [0.53657921, 0.89667129],
                    [0.14484776, 0.48805628],
                    [0.45985588, 0.0446123 ],
                    [0.95187448, 0.57575116],
                    [0.26455561, 0.77423369]])

    assert pex.n_points == n_points
    assert np.allclose(ref, pex.x)
