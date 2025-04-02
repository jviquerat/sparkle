import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.fpd import FPD
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
    s = EnvSpaces(loc_space)

    # We test different number of points as fixed_poisson_disc may
    # have issues providing the exact number of required points
    pms.n_points = 2
    pex = FPD(s, pms)
    assert pex.n_points == 2

    pms.n_points = 10
    pex = FPD(s, pms)

    ref = np.array([[0.54067565, 0.99920111],
                    [0.98837535, 0.03974221],
                    [0.0288999 , 0.09528166],
                    [0.02371881, 0.65633885],
                    [0.98779586, 0.70795355],
                    [0.55864886, 0.27985214],
                    [0.18618405, 0.96861411],
                    [0.27331024, 0.34363014],
                    [0.97949503, 0.42973387],
                    [0.99545743, 0.98057752]])

    assert pex.n_points == 10
    assert np.allclose(ref, pex.x)

    pms.n_points = 100
    pex = FPD(s, pms)
    assert pex.n_points == 100
