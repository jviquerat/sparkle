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

        self.dim_ = 1

    def covariance(self,
                   xi: ndarray,
                   xj: ndarray,
                   theta: ndarray) -> ndarray:
        """
        Computes the Matern 5/2 covariance function between two sets of points.

        Args:
            xi: The first set of points.
            xj: The second set of points.
            theta: The kernel parameters.

        Returns:
            The Matern 5/2 covariance matrix between xi and xj.
        """

        dist = distance_all_to_all(xi, xj)
        v0   = (1.0 + math.sqrt(5.0)*dist/theta[0] + 5.0/(3.0*theta[0])*dist**2)
        v    = v0*np.exp(-math.sqrt(5.0)*dist/theta[0])

        return v
