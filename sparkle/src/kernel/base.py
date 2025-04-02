import numpy as np
from   numpy import ndarray, matmul
from   numpy.linalg import solve
from typing import Optional

from sparkle.src.agent.ms_lbfgsb import MSLBFGSB
from sparkle.src.utils.distances import min_max_distance
from sparkle.src.env.spaces import EnvSpaces

###############################################
### Base kernel
class BaseKernel():
    def __init__(self, spaces: EnvSpaces) -> None:

        self.spaces = spaces
        self.reset()

    def reset(self) -> None:

        self.x0_       = None
        self.xmin_     = None
        self.xmax_     = None
        self.theta_    = None
        self.diag_eps_ = 1.0e-15

    @property
    def theta(self) -> None:
        return self.theta_

    def optimize(self, x: ndarray, y: ndarray) -> None:

        # Copy (x,y)
        self.x_  = x
        self.y_  = y
        self.ns_ = x.shape[0] # nb of samples
        self.nf_ = x.shape[1] # nb of features

        # Update bounds
        dmin, dmax = min_max_distance(self.x_)
        self.xmin_ = np.log(np.array([max(dmin,0.1)]))
        self.xmax_ = np.log(np.array([dmax]))

        # Optimize
        opt      = MSLBFGSB()
        x_opt, c = opt.optimize(self.log_likelihood,
                                self.xmin_,
                                self.xmax_,
                                n_pts=10*self.dim_)

        self.theta_ = np.exp(x_opt)

    # Compute log-likelihood
    def log_likelihood(self, log_theta: ndarray) -> float:

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
    def __call__(self,
                 xi: ndarray,
                 xj: ndarray,
                 theta: Optional[ndarray]=None) -> ndarray:

        if theta is None: theta = self.theta_

        K = self.covariance(xi, xj, theta)
        np.fill_diagonal(K, K.diagonal() + self.diag_eps_)

        return K
