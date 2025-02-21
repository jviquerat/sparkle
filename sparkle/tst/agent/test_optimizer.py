# Generic imports
import pytest
import numpy as np

# Custom imports
from sparkle.tst.tst             import set_seeds
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.utils.compare   import compare
from sparkle.src.env.spaces      import environment_spaces

###############################################
### Test transparent optimizer
def test_optimizer():

    # Set seed for reproducible test
    set_seeds(0)

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

    loc_space = {"dim": dim, "x0": x0, "xmin": xmin, "xmax": xmax}
    s    = environment_spaces(loc_space)
    opt  = optimizer(name, s, n_points, n_steps_max, parabola)
    x, c = opt.optimize()

    assert(compare(x[0],  0.001224214604398248,   1.0e-15))
    assert(compare(x[1], -0.00024287762152097391, 1.0e-15))
    assert(compare(c,     5.768116382852788e-08,  1.0e-15))
