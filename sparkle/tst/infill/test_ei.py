import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.infill.ei import EI
from sparkle.src.model.kriging import Kriging
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_ei():

    # Set seed for reproducible test
    set_seeds(0)

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([1,1])}
    space        = EnvSpaces(space_dict)
    lhs_pex      = LHS(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    pms             = types.SimpleNamespace()
    pms.kernel      = types.SimpleNamespace()
    pms.kernel.name = "matern52"
    model           = Kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    k  = np.argmin(y)
    xb = lhs_pex.x[k]
    yb = y[k]

    inf = EI(space, model)
    inf.set_best(xb, yb)
    x = np.array([[0.5,0.5]])
    vei = inf(x)
    ref = np.array([0.0])

    assert np.allclose(vei, ref)

    x = np.array([[0.5,0.5],
                  [0.2,0.2]])
    vei = inf(x)
    ref = np.array([0.0, 0.0])
    assert np.allclose(vei, ref)
