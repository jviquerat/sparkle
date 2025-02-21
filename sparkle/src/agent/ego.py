# Generic imports
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
    def __init__(self, path, spaces, pms):
        super().__init__(path, spaces, pms)

        self.name             = "EGO"
        self.spaces           = spaces
        self.n_steps_max      = set_default("n_steps_max", 20, pms)
        self.recompute_theta_ = set_default("recompute_theta", False, pms)
        self.load_model_      = set_default("load_model", False, pms)
        self.n_points         = 1

        if (self.load_model_):
            if not hasattr(pms, "model_file"):
                error("ego", "__init__",
                      "load_model option requires model_file parameter")
            self.model_file_ = pms.model_file

        self.model = kriging(spaces)

        self.summary()

    # Reset
    def reset(self, run):

        super().reset(run)
        self.model.reset()

    # Build initial model
    def build_initial_model(self, x, y):

        self.x_ = x.copy()
        self.y_ = y.copy()

        self.model.build(self.x_, self.y_, True)

        spacer()
        print("Built initial model")
        self.post_model_print()

    # Update model with new points
    def update_model(self, x, y):

        self.x_ = np.vstack((self.x_, x))
        self.y_ = np.hstack((self.y_, y))

        self.model.build(self.x_, self.y_, self.recompute_theta_)

    # Load saved model
    def load_model(self):

        self.model.load(self.model_file_)
        self.x_ = self.denormalize(self.model.x_)
        self.y_ = self.model.y_

        spacer()
        print("Loaded initial model")
        self.post_model_print()

    # Print after building or loading model
    def post_model_print(self):

        xb, yb = self.best_point()
        gs     = f"{yb:.5e}"
        gb     = np.array2string(xb, precision=5,
                                 floatmode='fixed', threshold=4, separator=',')

        spacer()
        print("Best initial score = "+str(gs)+" for x = "+str(gb))

    # Dump model
    def dump_model(self, filename):

        self.model.dump(filename)

    # Return best point
    def best_point(self):

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

        self.update_model(x, c)
        self.stp += 1

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max): return True
        return False

    # Denormalize inputs
    def denormalize(self, x):

        xx = self.spaces.xmin + (self.spaces.xmax - self.spaces.xmin)*x
        return xx
