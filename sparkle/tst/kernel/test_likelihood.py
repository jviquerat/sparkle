import os
import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.gaussian import Gaussian
from sparkle.src.kernel.matern52 import Matern52
from sparkle.src.pex.lhs import LHS


###############################################
def test_likelihood():

    pms          = types.SimpleNamespace()
    pms.n_points = 4
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([1,1])}
    space        = EnvSpaces(space_dict)
    lhs_pex      = LHS(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    # Test likelihood gradient with gaussian kernel
    kernel = Gaussian(space)
    kernel.optimize(lhs_pex.x, y)

    log_theta = np.log(np.array([0.5]))
    dLdt = kernel.grad_log_likelihood(log_theta)

    eps = 1.0e-8
    log_theta_plus = log_theta + eps
    log_theta_minus = log_theta - eps
    dLdt_fd = np.array([kernel.log_likelihood(log_theta_plus) -
                        kernel.log_likelihood(log_theta_minus)])/(2.0*eps)

    assert np.allclose(dLdt, dLdt_fd)

    # Test likelihood gradient with matern52 kernel
    kernel = Matern52(space)
    kernel.optimize(lhs_pex.x, y)

    log_theta = np.log(np.array([0.5, 0.5]))
    dLdt = kernel.grad_log_likelihood(log_theta)

    eps = 1.0e-8
    dx = np.array([eps, 0.0])
    dy = np.array([0.0, eps])
    log_theta_plus_dx = log_theta + dx
    log_theta_minus_dx = log_theta - dx
    log_theta_plus_dy = log_theta + dy
    log_theta_minus_dy = log_theta - dy

    dLdt_fd = np.array([kernel.log_likelihood(log_theta_plus_dx) -
                        kernel.log_likelihood(log_theta_minus_dx),
                        kernel.log_likelihood(log_theta_plus_dy) -
                        kernel.log_likelihood(log_theta_minus_dy)])/(2.0*eps)
    dLdt_fd = dLdt_fd.reshape(dLdt.shape)

    assert np.allclose(dLdt, dLdt_fd)
