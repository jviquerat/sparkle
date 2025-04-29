import os
import types

import numpy as np
import pytest

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.model.lipnet import LipNet
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_lipnet():

    set_seeds(0)

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([2,3])}
    space        = EnvSpaces(space_dict)
    lhs_pex      = LHS(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    pms               = types.SimpleNamespace()
    pms.ensemble_size = 5
    pms.n_epochs_max  = 20000
    pms.target_loss   = 1.0e-4
    pms.lr            = 5.0e-4
    pms.beta          = 0.0
    model             = LipNet(space, ".", pms)
    model.build(lhs_pex.x, y)

    # Test evaluate() function
    x_new = np.array([[0.5,0.5]])
    y_new = model.evaluate(x_new)

    # Test evaluate_grad() function
    x_test = np.array([[0.5, 0.5],
                       [0.2, 0.8]])
    grad_mu, grad_std = model.evaluate_grad(x_test)
    grad_mu_fd = np.zeros_like(grad_mu)
    grad_std_fd = np.zeros_like(grad_std)
    eps = 1.0e-8

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

    # XXX The standard deviation calculation involves sigma=sqrt(variance)
    # The gradient calculation via autograd needs to differentiate through
    # this square root. When the variance across the ensemble members is very
    # small (i.e., the models agree closely, for example in the vicinity of
    # samples), the argument inside the square root is close to zero.
    # Gradients involving sqrt(x) (which behave like 1/x) can become numerically
    # unstable or have high error when x is near zero.
    # Hence, the tolerances for the grad_std are very low
    assert np.allclose(grad_mu, grad_mu_fd)
    assert np.allclose(grad_std, grad_std_fd, rtol=1.0e-2, atol=1.0e-1)

    filename = "loss.png"
    os.remove(filename)
