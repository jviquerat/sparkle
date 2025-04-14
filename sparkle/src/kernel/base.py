from typing import Optional

import numpy as np
from numpy import matmul, ndarray
from numpy.linalg import solve, cholesky, slogdet

from sparkle.src.agent.ms_lbfgsb import MSLBFGSB
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.distances import min_max_distance_in_set


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
        self.diag_eps_ = 1.0e-10

    @property
    def theta(self) -> None:
        """
        Returns the kernel parameters.
        """
        return self.theta_

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

    def optimize(self, x: ndarray, y: ndarray) -> None:
        """
        Optimizes the kernel parameters based on the given data by
        maximizing log-likelihood

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
        dmin, dmax = min_max_distance_in_set(self.x_)
        self.xmin_ = np.log(max(dmin,0.1)*np.ones(self.dim_))
        self.xmax_ = np.log(dmax*np.ones(self.dim_))

        # Maximize log-likelihood
        f_lambda = lambda log_theta: -self.log_likelihood(log_theta)
        grad_lambda = lambda log_theta: -self.grad_log_likelihood(log_theta)
        opt      = MSLBFGSB()
        x_opt, c = opt.optimize(f_lambda,
                                self.xmin_,
                                self.xmax_,
                                df=grad_lambda,
                                n_pts=10*self.dim_,
                                m=20,
                                tol=1.0e-6,
                                max_iter=200)

        self.theta_ = np.exp(x_opt)

    def log_likelihood(self, log_theta: ndarray) -> float:
        """
        Computes the log-likelihood of the data given the log
        of the kernel parameters:

        log_lkh = -0.5*( res.T @ K_inv @ res + log(|K|) + n*log(2*pi) )

        Args:
            log_theta: The logarithm of the kernel parameters.

        Returns:
            The log-likelihood value.
        """

        theta = np.exp(log_theta)
        K = self(self.x_, self.x_, theta)
        L = cholesky(K)
        res = self.y_ - np.mean(self.y_)

        weighted_variance = np.inner(res.T, self.solve_linsys(L, res))
        const = np.log(2.0*np.pi)*self.ns_
        sgn, logabsdet = slogdet(K)
        log_lkh = -0.5*(weighted_variance + logabsdet + const)

        return log_lkh

    def grad_log_likelihood(self, log_theta: ndarray) -> float:
        """
        Computes the gradient of the log-likelihood w.r.t. the
        log of the kernel parameters:

        The derivation exploits:
            - the gradient of the inverse of a matrix
            - the gradient of the log-determinant
            - the invariance of trace by circular permutation

        dL/dlog_theta = dL/dtheta * dtheta/dlog_theta

        with

        dtheta/dlog_theta = theta
        dL/dtheta = -0.5*tr((K_inv - res @ res.T) @ dK/dtheta)

        where res = (y - mu)

        Args:
            log_theta: The logarithm of the kernel parameters.

        Returns:
            The gradient of the log-likelihood
        """

        theta = np.exp(log_theta)
        n = self.ns_
        p = len(theta)
        dK_dtheta = self.covariance_dtheta(self.x_, self.x_, theta)

        K     = self(self.x_, self.x_, theta)
        L     = cholesky(K)
        res   = self.y_ - np.mean(self.y_)
        alpha = self.solve_linsys(L, res)
        K_inv = self.solve_linsys(L, np.eye(n))

        grad = np.zeros(p)
        for i in range(p):
            M       = K_inv - np.outer(alpha, alpha.T)
            grad[i] = -0.5*np.trace(M @ dK_dtheta[:,:,i])

        return grad*theta

    def solve_linsys(self, L: ndarray, b: ndarray) -> ndarray:
        """
        Solves the linear system K * x = b using Cholesky decomposition.
        It first solves L*y = b, then L.T*x = y

        Args:
            L: cholesky matrix
            b: right hand side vector

        Returns:
            x: solution of the linear system
        """

        return solve(L.T, solve(L, b))

    def covariance(self,
                   x: ndarray,
                   y: ndarray,
                   theta: ndarray) -> ndarray:
        """
        Computes the Gaussian covariance matrix between two sets of points
        """

        raise NotImplementedError

    def covariance_dx(self,
                      x: ndarray,
                      y: ndarray,
                      theta: ndarray) -> ndarray:
        """
        Computes the derivative of the Gaussian covariance function with
        respect to the first variable x
        """

        raise NotImplementedError

    def covariance_dtheta(self,
                          x: ndarray,
                          y: ndarray,
                          theta: ndarray) -> ndarray:
        """
        Computes the derivative of the Gaussian covariance function with
        respect to the parameters
        """

        raise NotImplementedError
