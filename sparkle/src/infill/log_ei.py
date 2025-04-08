from math import erf, exp, expm1, log, log1p, pi, sqrt
from typing import Any

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces


###############################################
class LogEI():
    """
    Log Expected Improvement (LogEI) infill criterion.

    This class implements the Log Expected Improvement infill criterion, a
    variant of the Expected Improvement that uses a logarithmic transformation
    to improve numerical stability. It is used in Bayesian optimization to
    select the next point to evaluate.
    """
    def __init__(self, spaces: EnvSpaces, model: Any) -> None:
        """
        Initializes the LogEI infill criterion.

        Args:
            spaces: The environment's search space definition.
            model: The surrogate model (e.g., Kriging) used for function approximation.
        """

        self.spaces = spaces
        self.model  = model
        self.bound  =-1.0e8

    def set_best(self, xb: ndarray, yb: float) -> None:
        """
        Sets the current best observation.

        Args:
            xb: The best point found so far.
            yb: The function value at the best point.
        """

        self.xb = xb
        self.yb = yb

    def _log_ei(self, x: ndarray) -> ndarray:
        """
        Computes the Log Expected Improvement at a set of points.

        Args:
            x: A NumPy array of points at which to compute the LogEI.

        Returns:
            A NumPy array of the LogEI values at the given points.
        """

        x       = np.reshape(x, (-1,self.spaces.dim))
        mu, std = self.model.evaluate(x)
        std     = np.maximum(std, 1e-8)

        n    = x.shape[0]
        lgei = np.zeros(n)
        c1   = 0.5*log(2.0*pi)
        c2   = 0.5*log(0.5*pi)
        for i in range(n):
            z = (self.yb - mu[i])/std[i]
            if (z > -1.0):
                phi     = (1.0/sqrt(2.0*pi))*exp(-0.5*z**2)
                Phi     = 0.5*(1.0 + erf(z/sqrt(2.0)))
                lgei[i] = log(phi + z*Phi)
            elif (z > self.bound):
                v       = erfcx(-z/sqrt(2.0))
                v       = log(v*abs(z)) + c2
                lgei[i] = -0.5*z**2 - c1 + log1mexp(v)
            else:
                lgei[i] = -0.5*z**2 - c1 - 2.0*log(abs(z))

            lgei[i] += log(std[i])

        return lgei

    def __call__(self, x: ndarray) -> ndarray:
        """
        Computes the Log Expected Improvement at a set of points.

        This method is an alias for _log_ei, allowing the LogEI object to be
        called as a function.

        Args:
            x: A NumPy array of points at which to compute the LogEI.

        Returns:
            A NumPy array of the LogEI values at the given points.
        """

        return self._log_ei(x)

def log1mexp(x: float) -> float:
    """
    Numerically stable version of log(1 - exp(x)).

    Args:
        x: The input value.

    Returns:
        The result of log(1 - exp(x)).
    """

    if (x > -log(2)):
        return log(-expm1(x))
    else:
        return log1p(-exp(x))

def erfcx(x: float) -> float:
    """
    Approximation of the scaled complementary error function.

    This function uses an approximation from:
    "Closed-form approximations to the error and complementary error
    functions and their applications in atmospheric science", Ren et al (2007)

    Args:
        x: The input value.

    Returns:
        The approximate value of erfcx(x).
    """

    a = 2.9110
    v = a/((a-1.0)*sqrt(pi*x*x) + sqrt(pi*x*x + a*a))

    return v
