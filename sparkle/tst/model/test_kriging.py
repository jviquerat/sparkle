import os
import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.model.kriging import Kriging
from sparkle.src.pex.lhs import LHS


###############################################
def test_kriging():

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([2,3])}
    space        = EnvSpaces(space_dict)
    lhs_pex      = LHS(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    pms             = types.SimpleNamespace()
    pms.kernel      = types.SimpleNamespace()
    pms.kernel.name = "gaussian"
    model           = Kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    # Test evaluate() function
    x_new = np.array([[0.5,0.5]])
    y_new = model.evaluate(x_new)

    filename = "kriging_test.dat"
    model.dump(filename)
    model.load(filename)

    assert np.allclose(model.evaluate(x_new), y_new)
    os.remove(filename)

    # Test normalization
    x_test = np.array([[0.5, 0.5],
                       [0.2, 0.8]])
    x_transform = model.denormalize(model.normalize(x_test))
    assert np.allclose(x_test, x_transform)

    # Test evaluate_grad() function
    grad_mu, grad_std = model.evaluate_grad(x_test)
    grad_mu_fd = np.zeros_like(grad_mu)
    grad_std_fd = np.zeros_like(grad_std)
    eps = 1.0e-6

    for i in range(x_test.shape[0]):  # Loop over test points
        for j in range(space_dict["dim"]):  # Loop over dimensions
            dx = np.zeros_like(x_test[i])
            dx[j] = eps

            x_plus = x_test[[i]] + dx
            x_minus = x_test[[i]] - dx
            mu_plus, _ = model.evaluate(x_plus)
            mu_minus, _ = model.evaluate(x_minus)
            grad_mu_fd[i, j] = (mu_plus[0] - mu_minus[0])/(2.0 * eps)

            _, std_plus = model.evaluate(x_plus)
            _, std_minus = model.evaluate(x_minus)
            grad_std_fd[i, j] = (std_plus[0] - std_minus[0])/(2.0 * eps)

    assert np.allclose(grad_mu, grad_mu_fd)
    assert np.allclose(grad_std, grad_std_fd)
