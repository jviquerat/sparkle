# Generic imports
import numpy as np

# Local factory
from sparkle.src.core.factory import factory
from sparkle.src.agent.pso    import pso
from sparkle.src.agent.cmaes  import cmaes
from sparkle.src.agent.cem    import cem
from sparkle.src.agent.pbo    import pbo

local_factory = factory()

local_factory.register("pso",   pso)
local_factory.register("cmaes", cmaes)
local_factory.register("cem",   cem)
local_factory.register("pbo",   pbo)

###############################################
### The optimizer class is a transparent class for calling
### optimizers with a simple cost function, without the need
### to write a full environment
### It does not comply to the other agents interface
### It is assumed to run sequentially
class optimizer():
    def __init__(self, name, spaces, pms, cost):

        # Initialize agent
        self.agent = local_factory.create(name, path=".", spaces=spaces, pms=pms)

        # Cost function
        self.cost = cost

        # Initial reset
        self.reset(0)

    # Reset
    def reset(self, run):

        self.agent.reset(run)

    # Perform optimization
    def optimize(self):

        # Loop until done
        best_x     = None
        best_score = 1.0e15
        while (not self.agent.done()):

            x = self.agent.sample()

            n_pts = x.shape[0]
            c     = np.zeros(n_pts)
            for p in range(n_pts):
                res = self.cost(x[p])
                if (hasattr(res, "__len__")):
                    c[p] = res[0]
                else:
                    c[p] = res

                if (c[p] < best_score):
                    best_score = c[p]
                    best_x     = x[p]

            self.agent.step(x, c)

        return best_x, best_score
