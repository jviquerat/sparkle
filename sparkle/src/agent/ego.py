# Generic imports
import numpy as np
from math import sqrt, pi, exp, erf

# Custom imports
from sparkle.src.utils.default   import set_default
from sparkle.src.agent.base      import base_agent
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.env.spaces      import environment_spaces
from sparkle.src.pex.pex         import pex_factory
from sparkle.src.model.kriging   import kriging
from sparkle.src.utils.prints    import spacer
from sparkle.src.utils.error     import error

###############################################
### EGO
class ego(base_agent):
    def __init__(self, path, spaces, pms):
        super().__init__(path, spaces, pms)

        self.name             = "EGO"
        self.spaces           = spaces
        self.n_steps_max      = set_default("n_steps_max", 20, pms)
        self.recompute_theta_ = set_default("recompute_theta", False, pms)

        self.load_model_      = set_default("load_model", False, pms)
        if (self.load_model_):
            if not hasattr(pms, "model_file"):
                error("ego", "__init__",
                      "load_model option requires model_file parameter")
            self.model_file_ = pms.model_file

        self.model = kriging(spaces)
        self.pex   = pex_factory.create(pms.pex.name,
                                        spaces = spaces,
                                        pms    = pms.pex)

        self.n_points      = 1
        self.n_steps_total = self.pex.n_points + self.n_steps_max

        if (not self.silent):
            self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        self.cnt = 0

        self.pex.reset()
        self.model.reset()

        self.x_        = None
        self.y_        = None
        self.is_built_ = False

    # Build initial kriging model
    def build_model(self, x=None, y=None):

        # Copy pex points and associated costs if this is the first call
        # Otherwise, update the x and y vectors and rebuild model
        if (self.x_ is None):
            self.x_ = self.pex.x
            self.y_ = y

            self.model.build(self.x_, self.y_, True)
        else:
            self.x_ = np.vstack((self.x_, x))
            self.y_ = np.hstack((self.y_, y))

            self.model.build(self.x_, self.y_, self.recompute_theta_)

        if (not self.is_built_): self.finalize_initial_model()

    # Denormalize inputs
    def denormalize(self, x):

        xx = self.spaces.xmin + (self.spaces.xmax - self.spaces.xmin)*x
        return xx

    # Load saved model
    def load_model(self):

        self.model.load(self.model_file_)
        self.x_ = self.denormalize(self.model.x_)
        self.y_ = self.model.y_

        self.store(self.x_, self.y_)

        self.finalize_initial_model()

    # Dump model
    def dump_model(self, filename):

        self.model.dump(filename)

    # Wrap initial model
    def finalize_initial_model(self):

        if (not self.is_built_):
            self.is_built_ = True

            xb, yb = self.best()
            gs = f"{yb:.5e}"
            gb = np.array2string(xb, precision=5,
                                 floatmode='fixed', threshold=4, separator=',')

            spacer()
            if (self.load_model_): print("Loaded initial model")
            else: print("Built initial model")
            spacer()
            print("Best initial score = "+str(gs)+" for x = "+str(gb))

    # Return nb of points in pex
    def n_points_pex(self):

        return self.pex.n_points

    # Return i-th point of pex
    def pex_point(self, i):

        return self.pex.point(i)

    # Return best point
    def best(self):

        k = np.argmin(self.y_)

        return self.x_[k], self.y_[k]

    # Sample new point based on expected improvement
    def sample(self):

        name        = "cmaes"
        n_points    = 200
        n_steps_max = 10

        opt  = optimizer(name, self.spaces, n_points, n_steps_max, self.exp_imp)
        x, c = opt.optimize()
        x    = np.reshape(x, (-1,self.spaces.dim))

        return x

    # Compute expected improvement
    # We actually return -ei so it can be directly minimized
    def exp_imp(self, x):

        x       = np.reshape(x, (-1,self.dim))
        mu, std = self.model.evaluate(x)
        xb, yb  = self.best()

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

        self.store(x, c)
        self.build_model(x, c)

        self.stp += 1

    # Print informations
    def summary(self):

        super().summary()
        self.pex.summary()

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max):
            return True

        return False

    # Print informations
    def summary(self):

        spacer()
        print("Using "+self.name+" algorithm with "+str(self.n_steps_max)+" points")
        self.pex.summary()
        spacer()
        print("Problem dimensionality is "+str(self.dim))

    # Print
    def print(self):

        # Total nb of evaluations
        n_eval = self.pex.n_points + self.stp + 1

        # Handle no-printing after max step
        if (self.stp < self.n_steps_max-1):
            end = "\r"
            self.cnt = 0
        else:
            end  = "\n"
            self.cnt += 1

        # Actual print
        if (self.cnt <= 1):
            xb, yb = self.best()
            gs = f"{yb:.5e}"
            gb = np.array2string(xb, precision=5,
                                 floatmode='fixed', threshold=4, separator=',')

            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" for x = "+str(gb)+"                                                                                                           ", end=end)
