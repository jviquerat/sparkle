# Generic imports
import pytest
import numpy as np

# Custom imports
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.env.spaces      import environment_spaces
from sparkle.tst.tst             import *

###############################################
### Test transparent optimizer
def test_optimizer():

    # Parabola test function
    def parabola(x):

        v = 0.0
        for i in range(len(x)):
            v += (x[i])**2

        return v

    name        = "cmaes"
    dim         = 2
    x0          = 2.5*np.ones(dim)
    xmin        =-5.0*np.ones(dim)
    xmax        = 5.0*np.ones(dim)
    n_points    = 10
    n_steps_max = 30

    s    = environment_spaces([dim, x0, xmin, xmax], None)
    opt  = optimizer(name, s, n_points, n_steps_max, parabola)
    x, c = opt.optimize()

    assert(np.all(np.abs(x) < 1.0e-3))
    assert(c < 1.0e-6)
