import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.gaussian import Gaussian
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_gaussian():

    # Set seed for reproducible test
    set_seeds(0)

    space_dict = {"dim": 2,
                  "x0": None,
                  "xmin": np.array([0,0]),
                  "xmax": np.array([1,1])}
    space      = EnvSpaces(space_dict)

    # Check base kernel
    kernel = Gaussian(space)
    x0     = np.array([[0.5,0.5]])
    x1     = np.array([[0.6,0.6]])
    theta  = np.array([0.1, 0.2])

    K = kernel(x0, x1, theta)
    K_ref = np.array([[0.01471518]])
    assert np.allclose(K, K_ref)

    # Check shapes
    x0     = np.array([[0.5,0.5],
                       [0.4,0.4]])
    x1     = np.array([[0.6,0.6],
                       [0.1,0.1]])
    theta  = np.array([0.1, 0.2])

    K = kernel(x0, x1, theta)
    K_ref = np.array([[1.47151777e-02, 4.50140699e-09],
                      [7.32625556e-04, 4.93649216e-06]])
    assert np.allclose(K, K_ref)

    # Check derivative wrt x
    kernel = Gaussian(space)
    x0     = np.array([[0.5,0.5]])
    x1     = np.array([[0.6,0.6]])
    theta  = np.array([0.1, 0.2])

    K = kernel(x0, x1, theta)
    dKdx = kernel.covariance_dx(x0, x1, theta)

    eps = 1.0e-8
    dx = np.array([[eps, 0.0]])
    dy = np.array([[0.0, eps]])
    dKdx_fd = np.array([kernel(x0+dx, x1, theta) - kernel(x0-dx, x1, theta),
                        kernel(x0+dy, x1, theta) - kernel(x0-dy, x1, theta)])/(2.0*eps)
    dKdx_fd = dKdx_fd.reshape(dKdx.shape)

    assert np.allclose(dKdx, dKdx_fd)

    # Check derivative wrt theta
    dKdt = kernel.covariance_dtheta(x0, x1, theta)

    eps = 1.0e-8
    dx = np.array([eps, 0.0])
    dy = np.array([0.0, eps])
    dKdt_fd = np.array([kernel(x0, x1, theta+dx) - kernel(x0, x1, theta-dx),
                        kernel(x0, x1, theta+dy) - kernel(x0, x1, theta-dy)])/(2.0*eps)
    dKdt_fd = dKdt_fd.reshape(dKdt.shape)

    print(dKdt)
    print(dKdt_fd)

    assert np.allclose(dKdt, dKdt_fd)
