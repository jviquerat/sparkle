# Generic imports
import os
import math
import pytest
import types
import numpy as np

# Custom imports
from sparkle.src.utils.seeds     import set_seeds
from sparkle.src.kernel.gaussian import gaussian
from sparkle.src.env.spaces      import env_spaces

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

    kernel = gaussian(space)
    x0     = np.array([[0.5,0.5]])
    x1     = np.array([[0.6,0.6]])
    theta  = np.array([0.1])

    K = kernel(x0, x1, theta)
    K_ref = np.array([[0.3678794411714425]])
    assert np.allclose(K, K_ref)

    x0     = np.array([[0.5,0.5],
                       [0.4,0.4]])
    x1     = np.array([[0.6,0.6],
                       [0.1,0.1]])
    theta  = np.array([0.1])

    K = kernel(x0, x1, theta)
    K_ref = np.array([[3.67879441e-01, 1.12535175e-07],
                      [1.83156389e-02, 1.23409804e-04]])
    assert np.allclose(K, K_ref)
