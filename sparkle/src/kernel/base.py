# Generic imports
import types
import numpy as np
from   numpy import matmul
from   numpy.linalg import solve

# Custom imports
from sparkle.src.agent.ms_lbfgsb import ms_lbfgsb

###############################################
### Base kernel
class base_kernel():
    def __init__(self, spaces):

        self.spaces = spaces
        self.reset()

    def reset(self):

        self.x0_       = None
        self.xmin_     = None
        self.xmax_     = None
        self.theta_    = None
        self.diag_eps_ = 1.0e-15

    @property
    def theta(self):
        return self.theta_

    def optimize(self, x, y):

        # Copy (x,y)
        self.x_  = x
        self.y_  = y
        self.ns_ = x.shape[0] # nb of samples
        self.nf_ = x.shape[1] # nb of features

        # Update bounds
        dmin, dmax = self.distances(self.x_)
        self.xmin_ = np.log(np.array([max(dmin,0.1)]))
        self.xmax_ = np.log(np.array([dmax]))

        # Optimize
        opt  = ms_lbfgsb()
        x, c = opt.optimize(self.log_likelihood,
                            self.xmin_,
                            self.xmax_,
                            n_pts=10*self.dim_)

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
    # x and y have shapes (n_batch, dim)
    def __call__(self, xi, xj, theta=None):

        if theta is None: theta = self.theta_

        K = self.covariance(xi, xj, theta)

        return K

    # Compute dmin and dmax on sample set
    def distances(self, x):

        dmin = 1.0e8
        dmax = 0.0
        for k in range(x.shape[0]):
            for l in range(k+1, x.shape[0]):
                d = np.linalg.norm(x[k] - x[l])
                if (d < dmin): dmin = d
                if (d > dmax): dmax = d

        return dmin, dmax
