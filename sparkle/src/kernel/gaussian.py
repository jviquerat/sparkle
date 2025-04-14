from types import SimpleNamespace
from typing import Optional

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.base import BaseKernel
from sparkle.src.utils.distances import pairwise_distances


class Gaussian(BaseKernel):
    """
    Isotropic Gaussian kernel.

    This class implements the Isotropic Gaussian (also known as Radial Basis
    Function or RBF) kernel, a commonly used kernel in Gaussian process
    regression. It is characterized by its infinite smoothness and is often
    used for modeling smooth functions.
    """
    def __init__(self,
                 spaces: EnvSpaces,
                 pms: Optional[SimpleNamespace]=None) -> None:
        """
        Initializes the Gaussian kernel.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the kernel.
        """
        super().__init__(spaces)

        self.dim_ = 2

    def covariance(self,
                   x: ndarray,
                   y: ndarray,
                   theta: Optional[ndarray]=None) -> ndarray:
        """
        Computes the Gaussian covariance matrix between two sets of points:

        k(x,y) = exp(-0.5*||x - y||²/theta²)

        where ||.|| is the euclidian distance between two points

        Args:
            x: The first set of points, shape (ni, d)
            y: The second set of points, shape (nj, d)
            theta: The kernel parameters.

        Returns:
            The Gaussian covariance matrix between x and y, shape (ni, nj)
        """
        if theta is None: theta = self.theta_

        sigma = theta[0]
        delta = theta[1]
        dist = pairwise_distances(x, y)
        v    = delta**2*np.exp(-0.5*(dist/sigma)**2)

        return v

    def covariance_dx(self,
                      x: ndarray,
                      y: ndarray,
                      theta: Optional[ndarray]=None) -> ndarray:
        """
        Computes the derivative of the Gaussian covariance function with
        respect to the first variable x:

        dk/dx(x,y) = -(x - y)*k(x, y)/theta²

        Args:
            x: The first set of points, shape (ni, d)
            y: The second set of points, shape (nj, d)
            theta: The kernel parameters.

        Returns:
            The derivative of the Gaussian covariance matrix with respect to x,
            with shape (ni, nj, d)
        """
        if theta is None: theta = self.theta_

        sigma = theta[0]
        delta = theta[1]
        dx = x[:, np.newaxis, :] - y[np.newaxis, :, :] # x - y
        dk = -(dx/sigma**2)*self.covariance(x, y, theta)[:, :, np.newaxis]

        return dk

    def covariance_dtheta(self,
                          x: ndarray,
                          y: ndarray,
                          theta: ndarray) -> ndarray:
        """
        Computes the derivative of the Gaussian covariance function with
        respect to the parameters

        dk/dtheta(x,y) = ||x-y||²*k(x, y)/theta³

        Args:
            x: The first set of points, shape (ni, d)
            y: The second set of points, shape (nj, d)
            theta: The kernel parameters, shape (m,)

        Returns:
            The derivative of the Gaussian covariance matrix with respect to theta,
            with shape (ni, nj, m)
        """

        # Here m = 2 (sigma, delta)
        sigma = theta[0]
        delta = theta[1]
        dk    = np.zeros((x.shape[0], y.shape[0], self.dim_))

        # Derivative w.r.t. sigma
        dist        = pairwise_distances(x, y)
        K           = self.covariance(x, y, theta)
        dk[:, :, 0] = (dist**2 / sigma**3)*K

        # Derivative w.r.t. delta
        dk[:, :, 1] = 2.0*delta*np.exp(-0.5*(dist/sigma)**2)

        return dk
