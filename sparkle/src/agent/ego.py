# Generic imports
import types
import numpy as np
from math import sqrt, pi, exp, erf

# Custom imports
from sparkle.src.utils.default   import set_default
from sparkle.src.agent.base      import base_agent
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.env.spaces      import environment_spaces
from sparkle.src.model.kriging   import kriging
from sparkle.src.utils.prints    import spacer
from sparkle.src.utils.error     import error

###############################################
### EGO
class ego(base_agent):
    def __init__(self, path, spaces, model, pms):
        super().__init__(path, spaces, pms)

        self.name        = "EGO"
        self.spaces      = spaces
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_points    = 1
        self.model       = model

        self.summary()

    # Reset
    def reset(self, run):

        super().reset(run)

    # Return best point
    def best_point(self):

        k = np.argmin(self.model.y)
        return self.model.x[k], self.model.y[k]

    # Sample new point based on expected improvement
    def sample(self):

        pms = types.SimpleNamespace()
        pms.n_points    = 200
        pms.n_steps_max = 100
        pms.clip        = True
        pms.silent      = True

        opt  = optimizer("cmaes", self.spaces, pms, self.exp_imp)
        x, c = opt.optimize()
        x    = np.reshape(x, (-1,self.spaces.dim))

        return x

    # Compute expected improvement
    # We actually return -ei so it can be directly minimized
    def exp_imp(self, x):

        x       = np.reshape(x, (-1,self.dim))
        mu, std = self.model.evaluate(x)
        xb, yb  = self.best_point()

        n  = x.shape[0]
        ei = np.zeros(n)
        for i in range(n):
            if std[i] < 1.0e-15:
                ei[i] = 0.0
            else:
                prob      = (yb - mu[i])/std[i]
                cum_dist  = 0.5*(1.0 + erf(prob/sqrt(2.0)))
                prob_dist = (1.0/sqrt(2.0*pi))*np.exp(-0.5*prob**2)
                ei[i]     = std[i]*(prob*cum_dist + prob_dist)

        return -ei

    # Step
    def step(self, x, c):

        self.stp += 1

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max): return True
        return False
