# Generic imports
import os
import math
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.utils.seeds     import set_seeds
from sparkle.src.kernel.matern52 import matern52
from sparkle.src.env.spaces      import env_spaces
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
    space      = env_spaces(space_dict)

    kernel = matern52(space)
    x0     = np.array([[0.5,0.5]])
    x1     = np.array([[0.6,0.6]])
    theta  = np.array([0.1])

    val = kernel.covariance(x0, x1, theta)
    assert(compare(val, 0.1902957050844289, 1.0e-15))
