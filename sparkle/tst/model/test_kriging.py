# Generic imports
import math
import pytest
import types
import numpy as np

# Custom imports
from sparkle.tst.tst           import *
from sparkle.src.model.kriging import kriging
from sparkle.src.pex.lhs       import lhs

###############################################
### Test kriging model
def test_kriging():

    # Create pex
    pms          = types.SimpleNamespace()
    pms.n_points = 10
    lhs_pex      = lhs(2, np.array([0,0]), np.array([1,1]), pms)
    y            = np.cos(lhs_pex.x()[:,0]) + np.cos(lhs_pex.x()[:,1])

    model = kriging()
    model.build(lhs_pex.x(), y)

    x_new = np.array([[0.5,0.5]])
    y_new = model.evaluate(x_new)

    filename = "kriging_test.dat"
    model.dump(filename)
    model.load(filename)

    assert(model.evaluate(x_new) == y_new)


