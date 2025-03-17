# Generic imports
import os
import math
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.model.kriging import kriging
from sparkle.src.pex.lhs       import lhs
from sparkle.src.env.spaces    import env_spaces

###############################################
### Test kriging model
def test_kriging():

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([1,1])}
    space        = env_spaces(space_dict)
    lhs_pex      = lhs(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    pms             = types.SimpleNamespace()
    pms.kernel      = types.SimpleNamespace()
    pms.kernel.name = "matern52"
    model           = kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    x_new = np.array([[0.5,0.5]])
    y_new = model.evaluate(x_new)

    filename = "kriging_test.dat"
    model.dump(filename)
    model.load(filename)

    assert(model.evaluate(x_new) == y_new)
    os.remove(filename)
