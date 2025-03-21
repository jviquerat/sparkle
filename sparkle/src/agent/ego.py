# Generic imports
import numpy as np

# Custom imports
from sparkle.src.utils.default   import set_default
from sparkle.src.agent.base      import base_agent
from sparkle.src.agent.ms_lbfgsb import ms_lbfgsb
from sparkle.src.infill.infill   import infill_factory
from sparkle.src.model.kriging   import kriging

###############################################
### EGO
class ego(base_agent):
    def __init__(self, path, spaces, model, pms):
        super().__init__(path, spaces, pms)

        self.name        = "EGO"
        self.spaces      = spaces
        self.n_points    = 1
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.model       = model

        # Initialize infill
        self.infill = infill_factory.create(pms.infill,
                                            spaces = spaces,
                                            model  = model)

        self.summary()

    # Reset
    def reset(self, run):

        super().reset(run)

    # Return best point
    def best_point(self):

        k = np.argmin(self.model.y)
        return self.model.x[k], self.model.y[k]

    # Local function for optimization of infill
    def opt_infill(self, x):

        return -self.infill(x)

    # Sample new point based on expected improvement
    def sample(self):

        # Set best point to infill before optimization
        xb, yb  = self.best_point()
        self.infill.set_best(xb, yb)

        # Optimize
        opt  = ms_lbfgsb()
        x, c = opt.optimize(self.opt_infill,
                            self.spaces.xmin,
                            self.spaces.xmax,
                            n_pts=20*self.spaces.dim,
                            tol=1.0e-6, max_iter=50)

        return np.reshape(x, (-1,self.spaces.dim))

    # Step
    def step(self, x, c):

        self.stp += 1

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max): return True
        return False
