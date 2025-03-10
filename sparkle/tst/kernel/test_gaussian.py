# Generic imports
import os
import math
import pytest
import types
import numpy as np

# Custom imports
from sparkle.tst.tst             import set_seeds
from sparkle.src.kernel.gaussian import gaussian
from sparkle.src.env.spaces      import environment_spaces
from sparkle.src.utils.compare   import compare

###############################################
### Test gaussian kernel
def test_gaussian():

    # Set seed for reproducible test
    set_seeds(0)

    space_dict = {"dim": 2,
                  "x0": None,
                  "xmin": np.array([0,0]),
                  "xmax": np.array([1,1])}
    space      = environment_spaces(space_dict)

    kernel = gaussian(space)
    x0     = np.array([[0.5,0.5]])
    x1     = np.array([[0.6,0.6]])
    theta  = np.array([0.1])

    val = kernel.covariance(x0, x1, theta)
    assert(compare(val, 0.3678794411714425, 1.0e-15))
