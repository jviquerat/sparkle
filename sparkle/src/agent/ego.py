# Generic imports
import numpy as np
from math import sqrt, pi, exp, erf

# Custom imports
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.pex.pex         import pex_factory
from sparkle.src.model.kriging   import kriging
from sparkle.src.utils.prints    import spacer

###############################################
### EGO
class ego():
    def __init__(self, path, dim, x0, xmin, xmax, pms):

        self.name        = "EGO"
        self.base_path   = path
        self.dim         = dim
        self.x0          = x0
        self.xmin        = xmin
        self.xmax        = xmax
        self.n_steps_max = pms.n_steps_max

        self.x_          = None
        self.y_          = None
        self.is_built_   = False

        self.pex = pex_factory.create(pms.pex.name,
                                      dim  = self.dim,
                                      xmin = self.xmin,
                                      xmax = self.xmax,
                                      pms  = pms.pex)
        self.model = kriging()

        self.n_steps_total = self.pex.n_points() + self.n_steps_max

        self.summary()

    # Build initial kriging model
    def build_model(self, x=None, y=None):

        # Copy pex points and associated costs if this is the first call
        # Otherwise, update the x and y vectors and rebuild model
        if (self.x_ is None):
            self.x_ = self.normalize(self.pex.x())
            self.y_ = y
        else:
            self.x_ = self.denormalize(self.x_)
            self.x_ = np.vstack((self.x_, x))
            self.y_ = np.hstack((self.y_, y))
            self.x_ = self.normalize(self.x_)

        # Build model
        self.model.build(self.x_, self.y_)

        # Initial screen output
        if (not self.is_built_):
            self.is_built_ = True

            xb, yb = self.best()
            gs = f"{yb:.5e}"
            gb = np.array2string(xb, precision=5, floatmode='fixed', threshold=4, separator=',')

            spacer()
            print("Built initial model")
            spacer()
            print("Best initial score = "+str(gs)+" for x = "+str(gb))

    # Return nb of points in pex
    def n_points_pex(self):

        return self.pex.n_points()

    # Return i-th point of pex
    def pex_point(self, i):

        return self.pex.point(i)

    # Return best point
    def best(self):

        k = np.argmin(self.y_)

        return self.x_[k], self.y_[k]

    # Normalize inputs
    def normalize(self, x):

        return (x - self.xmin)/(self.xmax - self.xmin)

    # Denormalize inputs
    def denormalize(self, x):

        return self.xmin + (self.xmax - self.xmin)*x

    # Reset
    def reset(self, run):

        self.stp = 0
        self.cnt = 0

        #self.pex.reset()
        self.model.reset()

    # Sample new point based on expected improvement
    def sample(self):

        name        = "cmaes"
        dim         = self.model.nf_
        x0          = 0.5*np.ones(dim)
        xmin        = np.zeros(dim)
        xmax        = np.ones(dim)
        n_points    = 200
        n_steps_max = 10

        opt  = optimizer(name, dim, x0, xmin, xmax,
                         n_points, n_steps_max, self.exp_imp)
        x, c = opt.optimize()
        x    = np.reshape(x, (-1,dim))

        return self.denormalize(x)

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
        n_eval = self.pex.n_points() + self.stp + 1

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
            gb = np.array2string(xb, precision=5, floatmode='fixed', threshold=4, separator=',')

            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" for x = "+str(gb)+"                                                                                                           ", end=end)

    # Dump data
    def dump(self):
        pass
