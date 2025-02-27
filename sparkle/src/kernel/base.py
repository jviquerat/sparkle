# Generic imports
import types
import numpy as np
from   numpy import matmul
from   numpy.linalg import solve

# Custom imports
from sparkle.src.env.spaces      import environment_spaces
from sparkle.src.agent.optimizer import optimizer

###############################################
### Base kernel
class base_kernel():
    def __init__(self, spaces):

        self.spaces = spaces

    @property
    def theta(self):
        return self.theta_

    def optimize(self, x, y):

        self.x_  = x
        self.y_  = y
        self.ns_ = x.shape[0] # nb of samples
        self.nf_ = x.shape[1] # nb of features

        dict_space = {"dim": self.dim_,
                      "x0": self.theta_,
                      "xmin": self.xmin_,
                      "xmax": self.xmax_}
        space      = environment_spaces(dict_space)

        pms             = types.SimpleNamespace()
        pms.n_points    = 200
        pms.n_steps_max = 10
        pms.clip        = True
        pms.silent      = True
        opt             = optimizer("cmaes", space, pms, self.log_likelihood)
        log_theta, cost = opt.optimize()

        self.theta_ = np.exp(log_theta)

    # Compute log-likelihood
    def log_likelihood(self, log_theta):

        theta = np.exp(log_theta)

        K_   = self(self.x_, self.x_, theta)
        ones = np.ones(self.ns_)
        mu   = matmul(ones.T, solve(K_, self.y_))
        mu  /= matmul(ones.T, solve(K_, ones))

        res     = self.y_ - mu
        var     = matmul(res.T, solve(K_, res))/self.ns_
        if (var < 0.0): return 1.0e10
        log_lkh =-0.5*(-self.ns_*np.log(var) + np.linalg.slogdet(K_)[1])

        return log_lkh

    # () operator
    def __call__(self, x, y, theta=None):

        if theta is None: theta = self.theta_

        nx = x.shape[0]
        ny = y.shape[0]
        K  = np.zeros((nx,ny))

        for i in range(nx):
            for j in range(ny):
                K[i,j] = self.covariance(x[i], y[j], theta)
                if (i==j): K[i,j] += self.diag_eps_

        return K
