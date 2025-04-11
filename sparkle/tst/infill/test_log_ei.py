import types

import pytest
import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.infill.log_ei import LogEI
from sparkle.src.model.kriging import Kriging
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.seeds import set_seeds

@pytest.mark.parametrize("kernel_type, ref0, ref1",
                         [("gaussian",
                           np.array([-77877.87462661]),
                           np.array([-77877.87463119, -734.27186907])),
                          ("matern52",
                           np.array([-2562.66880535]),
                           np.array([-2562.66880535, -77.39146467]))
                          ])
###############################################
def test_log_ei(kernel_type, ref0, ref1):

    # Set seed for reproducible test
    set_seeds(0)

    pms          = types.SimpleNamespace()
    pms.n_points = 10
    space_dict   = {"dim": 2,
                    "x0": None,
                    "xmin": np.array([0,0]),
                    "xmax": np.array([1,1])}
    space        = EnvSpaces(space_dict)
    lhs_pex      = LHS(space, pms)
    y            = np.cos(lhs_pex.x[:,0]) + np.cos(lhs_pex.x[:,1])

    pms             = types.SimpleNamespace()
    pms.kernel      = types.SimpleNamespace()
    pms.kernel.name = kernel_type
    model           = Kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    k  = np.argmin(y)
    xb = lhs_pex.x[k]
    yb = y[k]

    inf = LogEI(space, model)
    inf.set_best(xb, yb)

    # Test () function
    x = np.array([[0.5,0.5]])
    vei = inf(x)
    assert np.allclose(vei, ref0)

    x = np.array([[0.5,0.5],
                  [0.2,0.2]])
    vei = inf(x)
    assert np.allclose(vei, ref1)

    # # Test gradient function
    # grad_log_ei = inf.grad(x)
    # grad_log_ei_fd = np.zeros_like(grad_log_ei)
    # eps = 1.0e-8

    # for i in range(x.shape[0]):
    #     for j in range(space_dict["dim"]):
    #         dx = np.zeros_like(x[i])
    #         dx[j] = eps

    #         x_plus = x[[i]] + dx
    #         x_minus = x[[i]] - dx
    #         log_ei_plus = inf(x_plus)
    #         log_ei_minus = inf(x_minus)
    #         grad_log_ei_fd[i, j] = (log_ei_plus[0] - log_ei_minus[0])/(2.0 * eps)

    # assert np.allclose(grad_log_ei, grad_log_ei_fd)
