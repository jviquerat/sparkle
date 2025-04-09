import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.infill.ei import EI
from sparkle.src.model.kriging import Kriging
from sparkle.src.pex.lhs import LHS
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_ei():

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
    pms.kernel.name = "gaussian"
    model           = Kriging(space, ".", pms)
    model.build(lhs_pex.x, y)

    k  = np.argmin(y)
    xb = lhs_pex.x[k]
    yb = y[k]

    inf = EI(space, model)
    inf.set_best(xb, yb)

    # Test () function
    x = np.array([[0.5,0.5]])
    vei = inf(x)
    ref = np.array([0.0])

    assert np.allclose(vei, ref)

    x = np.array([[0.5,0.5],
                  [0.2,0.2]])
    vei = inf(x)
    ref = np.array([0.0, 0.0])
    assert np.allclose(vei, ref)

    # Test gradient function
    ei_grad = inf.ei_grad(x)
    ei_grad_fd = np.zeros_like(ei_grad)
    eps = 1.0e-8

    for i in range(x.shape[0]):
        for j in range(space_dict["dim"]):
            dx = np.zeros_like(x[i])
            dx[j] = eps

            x_plus = x[[i]] + dx
            x_minus = x[[i]] - dx
            ei_plus = inf(x_plus)
            ei_minus = inf(x_minus)
            ei_grad_fd[i, j] = (ei_plus[0] - ei_minus[0])/(2.0 * eps)

    assert np.allclose(ei_grad, ei_grad_fd)
