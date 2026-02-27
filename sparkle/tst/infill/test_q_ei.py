import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.infill.q_ei import QEI
from sparkle.src.model.kriging import Kriging
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.seeds import set_seeds


def test_q_ei():
    set_seeds(0)

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0, 0]),
                    "xmax": np.array([1, 1])}
    space        = EnvSpaces(space_dict)
    lhs_pex      = LHS(space, pms)
    y            = np.cos(lhs_pex.x[:, 0]) + np.cos(lhs_pex.x[:, 1])

    pms             = types.SimpleNamespace()
    pms.kernel      = types.SimpleNamespace()
    pms.kernel.name = "matern52"
    model           = Kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    k  = np.argmin(y)
    xb = lhs_pex.x[k]
    yb = y[k]

    inf = QEI(space, model)
    inf.set_best(xb, yb)

    # Test q=1 (should be close to standard EI)
    x = np.array([[0.5, 0.5]])
    v_qei = inf(x)
    assert v_qei.shape == (1,)

    # Test q=2 independent queries (2D array falls back to standard EI)
    x = np.array([[0.5, 0.5], [0.2, 0.2]])
    v_qei_2 = inf(x)
    assert v_qei_2.shape == (2,)

    # Test flattening behavior (used by MSLBFGSB for joint q-EI)
    x_flat = x.flatten()
    v_qei_flat = inf(x_flat)
    assert v_qei_flat.shape == (1,)
