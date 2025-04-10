import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.base import BaseKernel
from sparkle.src.utils.distances import distance_all_to_all


class Matern52(BaseKernel):
    """
    Isotropic Matern 5/2 kernel.

    This class implements the Matern 5/2 kernel, a commonly used kernel in
    Gaussian process regression. It is characterized by its smoothness and
    flexibility in modeling various types of functions.
    """
    def __init__(self,
                 spaces: EnvSpaces,
                 pms: Optional[SimpleNamespace]=None) -> None:
        """
        Initializes the Matern52 kernel.

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
        Computes the Matern 5/2 covariance function between two sets of points:

        k(x,y) = delta²*f(x,y)*g(x,y)
        f(x,y) = 1 + ratio*||x-y|| + ratio²*||x-y||²/3)
        g(x,y) = exp(-ratio*||x-y||)

        with ratio = sqrt(5)/sigma

        Args:
            x: The first set of points.
            y: The second set of points.
            theta: The kernel parameters.

        Returns:
            The Matern 5/2 covariance matrix between x and y.
        """
        if theta is None: theta = self.theta_

        dist  = distance_all_to_all(x, y)
        sigma = theta[0]
        delta = theta[1]
        ratio = math.sqrt(5.0)/sigma
        f    = (1.0 + ratio*dist + ratio**2*dist**2/3.0)
        g    = np.exp(-ratio*dist)

        return delta**2*f*g

    def covariance_dx(self,
                      x: ndarray,
                      y: ndarray,
                      theta: Optional[ndarray]=None) -> ndarray:
        """
        Computes the derivative of the Matern 5/2 covariance function with
        respect to the first variable x:

        dk/dx(x,y) = delta²*ratio*(x-y)*g(x,y)*((1 - f(x,y))/||x-y|| + 2*ratio/3)

        with ratio = sqrt(5)/sigma

        Args:
            x: The first set of points.
            y: The second set of points.
            theta: The kernel parameters.

        Returns:
            The derivative of the Matern 5/2 covariance matrix with respect to x.
        """
        if theta is None: theta = self.theta_
        
        dist = distance_all_to_all(x, y)
        dx   = x[:, np.newaxis, :] - y[np.newaxis, :, :] # x - y

        sigma = theta[0]
        delta = theta[1]
        ratio = math.sqrt(5.0)/sigma
        f     = (1.0 + ratio*dist + ratio**2*dist**2/3.0)
        g     = np.exp(-ratio*dist)

        v = ((1.0 - f)/dist + 2.0*ratio/3.0)*delta**2*ratio*g # shape (ni, nj)
        v = v[:, :, np.newaxis]*dx # shape (ni, nj, d)

        return v

    def covariance_dtheta(self,
                          x: ndarray,
                          y: ndarray,
                          theta: ndarray) -> ndarray:
        """
        Computes the derivative of the Matern 5/2 covariance function with
        respect to the parameters theta:

        dk/dsigma(x,y) = R*(K(x,y) - delta²*(1 + 2*ratio*||x-y||/3))*g(x,y))
        dk/ddelta(x,y) = 2*delta*K(x,y)

        with R     = ratio*||x-y||/sigma
        with ratio = sqrt(5)/sigma


        Args:
            x: The first set of points, shape (ni, d)
            y: The second set of points, shape (nj, d)
            theta: The kernel parameters, shape (m,)

        Returns:
            The derivative of the Gaussian covariance matrix with respect to theta,
            with shape (ni, nj, m)
        """

        # Here m = 2 (sigma, delta)
        m     = theta.shape[0]
        dk    = np.zeros((x.shape[0], y.shape[0], m))

        # Derivative w.r.t. sigma
        dist  = distance_all_to_all(x, y)
        sigma = theta[0]
        delta = theta[1]
        ratio = math.sqrt(5.0)/sigma
        R     = ratio*dist/sigma
        f     = (1.0 + ratio*dist + ratio**2*dist**2/3.0)
        g     = np.exp(-ratio*dist)
        K     = self.covariance(x, y, theta)

        dk[:, :, 0] = R*(K - delta**2*(1.0 + 2.0*ratio*dist/3.0)*g)

        # Derivative w.r.t. delta
        dk[:, :, 1] = 2.0*delta*f*g

        return dk
