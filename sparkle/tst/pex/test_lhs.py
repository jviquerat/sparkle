# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.pex.lhs     import LHS
from sparkle.src.env.spaces  import EnvSpaces
from sparkle.src.utils.seeds import set_seeds

###############################################
### Test lhs pex
def test_lhs():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 12

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)
    pex = LHS(s, pms)

    ref = np.array([[0.04573446, 0.7482182 ],
                    [0.14293245, 0.81659655],
                    [0.21689695, 0.48605165],
                    [0.29540693, 0.5648464 ],
                    [0.3686379 , 0.25726077],
                    [0.47049118, 0.3350182 ],
                    [0.5364656 , 0.17258634],
                    [0.65764775, 0.98171076],
                    [0.7469719 , 0.87178995],
                    [0.78195346, 0.16046639],
                    [0.89931042, 0.04733705],
                    [0.96074124, 0.65583435]])

    assert pex.n_points == n_points
    assert np.allclose(ref, pex.x)
