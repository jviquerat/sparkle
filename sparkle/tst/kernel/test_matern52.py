import numpy as np

from sparkle.src.utils.seeds     import set_seeds
from sparkle.src.kernel.matern52 import Matern52
from sparkle.src.env.spaces      import EnvSpaces

###############################################
### Test matern52 kernel
def test_matern52():

    # Set seed for reproducible test
    set_seeds(0)

    space_dict = {"dim": 2,
                  "x0": None,
                  "xmin": np.array([0,0]),
                  "xmax": np.array([1,1])}
    space      = EnvSpaces(space_dict)

    kernel = Matern52(space)
    x0     = np.array([[0.5,0.5]])
    x1     = np.array([[0.6,0.6]])
    theta  = np.array([0.1])

    K = kernel(x0, x1, theta)
    K_ref = np.array([[0.1902957050844289]])
    assert(np.allclose(K, K_ref))

    x0     = np.array([[0.5,0.5],
                       [0.4,0.4]])
    x1     = np.array([[0.6,0.6],
                       [0.1,0.1]])
    theta  = np.array([0.1])

    K = kernel(x0, x1, theta)
    K_ref = np.array([[1.90295705e-01, 6.09415049e-05],
                      [1.55128831e-02, 1.02289432e-03]])
    assert(np.allclose(K, K_ref))
