# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.utils.seeds     import set_seeds
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.utils.compare   import compare
from sparkle.src.env.spaces      import env_spaces

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

    pms             = types.SimpleNamespace()
    pms.n_points    = 10
    pms.n_steps_max = 30

    dict_space = {"dim":  2,
                  "x0":   2.5*np.ones(2),
                  "xmin":-5.0*np.ones(2),
                  "xmax": 5.0*np.ones(2)}
    space      = env_spaces(dict_space)
    opt        = optimizer("cmaes", space, pms, parabola)
    x, c       = opt.optimize()

    assert compare(x[0],  0.0005683938259276194,  1.0e-15)
    assert compare(x[1], -0.00018952556245307778, 1.0e-15)
    assert compare(c,     1.3003505428756697e-09, 1.0e-15)
