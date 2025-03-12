# Generic imports
import os
import math
import pytest
import types
import numpy as np

# Custom imports
from sparkle.tst.tst           import set_seeds
from sparkle.src.model.kriging import kriging
from sparkle.src.infill.log_ei import log_ei
from sparkle.src.pex.lhs       import lhs
from sparkle.src.env.spaces    import environment_spaces
from sparkle.src.utils.compare import compare

###############################################
### Test ei infill
def test_ei():

    # Set seed for reproducible test
    set_seeds(0)

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([1,1])}
    space        = environment_spaces(space_dict)
    lhs_pex      = lhs(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    pms             = types.SimpleNamespace()
    pms.kernel      = types.SimpleNamespace()
    pms.kernel.name = "matern52"
    model           = kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    k  = np.argmin(y)
    xb = lhs_pex.x[k]
    yb = y[k]

    inf = log_ei(space, model)
    inf.set_best(xb, yb)
    x = np.array([[0.5,0.5]])
    vei = inf(x)

    assert(compare(vei, 1.2090202324678856, 1.0e-15))
