from typing import Optional

import numpy as np
from numpy import matmul, ndarray
from numpy.linalg import solve

from sparkle.src.agent.ms_lbfgsb import MSLBFGSB
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.distances import min_max_distance


class BaseKernel():
    """
    Base class for kernel functions.

    This class defines the common interface for all kernel functions used in
    surrogate models. It provides methods for computing the kernel matrix,
    optimizing kernel parameters, and computing the log-likelihood.
    """
    def __init__(self, spaces: EnvSpaces) -> None:
        """
        Initializes the BaseKernel.

        Args:
            spaces: The environment's search space definition.
        """

        self.spaces = spaces
        self.reset()

    def reset(self) -> None:
        """
        Resets the kernel parameters and internal state.
        """

        self.x0_       = None
        self.xmin_     = None
        self.xmax_     = None
        self.theta_    = None
        self.diag_eps_ = 1.0e-15

    @property
    def theta(self) -> None:
        """
        Returns the kernel parameters.
        """
        return self.theta_

    def optimize(self, x: ndarray, y: ndarray) -> None:
        """
        Optimizes the kernel parameters based on the given data.

        Args:
            x: The input data points.
            y: The corresponding target values.
        """

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

    def log_likelihood(self, log_theta: ndarray) -> float:
        """
        Computes the log-likelihood of the data given the kernel parameters.

        Args:
            log_theta: The logarithm of the kernel parameters.

        Returns:
            The log-likelihood value.
        """

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

    def __call__(self,
                 xi: ndarray,
                 xj: ndarray,
                 theta: Optional[ndarray]=None) -> ndarray:
        """
        Computes the kernel matrix between two sets of points.

        Args:
            xi: The first set of points.
            xj: The second set of points.
            theta: Optional kernel parameters. If None, uses the optimized parameters.

        Returns:
            The kernel matrix between xi and xj.
        """

        if theta is None: theta = self.theta_

        K = self.covariance(xi, xj, theta)
        np.fill_diagonal(K, K.diagonal() + self.diag_eps_)

        return K

    def covariance(self,
                   xi: ndarray,
                   xj: ndarray,
                   theta: ndarray) -> ndarray:
        """
        Computes the covariance function between two sets of points.

        Args:
            xi: The first set of points.
            xj: The second set of points.
            theta: The kernel parameters.

        Returns:
            The covariance matrix between xi and xj.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
