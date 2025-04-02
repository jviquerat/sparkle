from types import SimpleNamespace
from typing import Optional

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.base import BaseKernel
from sparkle.src.utils.distances import distance_all_to_all


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

        self.dim_ = 1

    def covariance(self,
                   xi: ndarray,
                   xj: ndarray,
                   theta: ndarray) -> ndarray:
        """
        Computes the Gaussian covariance function between two sets of points.

        Args:
            xi: The first set of points.
            xj: The second set of points.
            theta: The kernel parameters.

        Returns:
            The Gaussian covariance matrix between xi and xj.
        """

        dist = distance_all_to_all(xi, xj)
        v    = np.exp(-0.5*(dist/theta[0])**2)

        return v
