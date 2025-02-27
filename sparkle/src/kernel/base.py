# Generic imports
import types
import numpy as np
from   numpy import matmul
from   numpy.linalg import solve

# Custom imports
from sparkle.src.env.spaces      import environment_spaces
from sparkle.src.agent.ms_lbfgsb import ms_lbfgsb

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

        opt  = ms_lbfgsb()
        x, c = opt.optimize(self.log_likelihood,
                            self.xmin_,
                            self.xmax_,
                            10*self.dim_)

        self.theta_ = np.exp(x)

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
