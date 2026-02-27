from math import erf, exp, pi, sqrt
from typing import Any

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces


###############################################
### Expected improvement infill
class EI():
    """
    Expected Improvement (EI) infill criterion.

    This class implements the Expected Improvement infill criterion, which
    is used in Bayesian optimization to select the next point to evaluate.
    It balances exploration and exploitation by choosing points that are
    likely to improve upon the current best observation.
    """
    def __init__(self, spaces: EnvSpaces, model: Any, pms: Any=None) -> None:
        """
        Initializes the EI infill criterion.

        Args:
            spaces: The environment's search space definition.
            model: The surrogate model (e.g., Kriging) used for function approximation.
        """

        self.spaces = spaces
        self.model  = model

    def set_best(self, xb: ndarray, yb: float) -> None:
        """
        Sets the current best observation.

        Args:
            xb: The best point found so far.
            yb: The function value at the best point.
        """

        self.xb = xb
        self.yb = yb

    def ei(self, x: ndarray) -> ndarray:
        """
        Computes the expected improvement at a set of points:

        ei(x) = std(x)*(phi(z) + z*Phi(z))

        with z = (y* - mu(x))/std(x)
             phi(z) is the pdf of the normal distribution
             Phi(z) is the cdf of the normal distribution

        Args:
            x: A NumPy array of points at which to compute the EI, shape (n, d)

        Returns:
            A NumPy array of the EI values at the given points, shape (n,)
        """

        x       = np.reshape(x, (-1,self.spaces.dim))
        mu, std = self.model.evaluate(x)
        std     = np.maximum(std, 1e-8)

        n  = x.shape[0]
        ei = np.zeros(n)
        for i in range(n):
            z     = (self.yb - mu[i])/std[i]
            phi   = (1.0/sqrt(2.0*pi))*exp(-0.5*z**2)
            Phi   = 0.5*(1.0 + erf(z/sqrt(2.0)))
            ei[i] = std[i]*(phi + z*Phi)

        return ei

    def grad(self, x: ndarray) -> ndarray:
        """
        Computes the gradient of EI at a set of points:

        grad_ei(x) = - grad_m(x) Phi(z) + grad_s(x) phi(z)

        Args:
            x: A NumPy array of points at which to compute the gradient of EI, shape (n,d)

        Returns:
            A NumPy array of the EI gradient values at the given points, shape (n, d)
        """

        x       = np.reshape(x, (-1,self.spaces.dim))
        mu, std = self.model.evaluate(x)
        grad_mu, grad_std = self.model.evaluate_grad(x)

        grad_ei = np.zeros_like(x)
        for i in range(x.shape[0]):
            z     = (self.yb - mu[i])/std[i]
            phi   = (1.0/sqrt(2.0*pi))*exp(-0.5*z**2)
            Phi   = 0.5*(1.0 + erf(z/sqrt(2.0)))
            grad_ei[i] = -grad_mu[i,:]*phi + grad_std[i,:]*Phi

        return grad_ei

    def __call__(self, x: ndarray) -> ndarray:
        """
        Computes the expected improvement at a set of points
        This is an alias for _ei, allowing the EI class to be called as a function

        Args:
            x: A NumPy array of points at which to compute the EI

        Returns:
            A NumPy array of the EI values at the given points
        """

        return self.ei(x)
