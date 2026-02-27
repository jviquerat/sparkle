from typing import Any

import numpy as np
from numpy import ndarray
from numpy.linalg import cholesky

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.infill.ei import EI
from sparkle.src.utils.default import set_default


###############################################
class QEI():
    """
    Multi-point expected improvement (q-EI) infill criterion

    This class implements the q-EI criterion using Monte Carlo integration
    with the reparameterization trick. It allows for sampling multiple
    points in parallel. The random samples are fixed per instantiation
    to ensure the function is deterministic, allowing for finite difference
    gradient approximation by the optimizer
    """
    def __init__(self,
                 spaces: EnvSpaces,
                 model: Any,
                 pms: Any=None) -> None:
        """
        Initializes the QEI infill criterion

        Args:
            spaces: The environment's search space definition
            model: The surrogate model used for function approximation
            pms: Optional parameter namespace
        """
        self.spaces     = spaces
        self.model      = model
        self.mc_samples = set_default("mc_samples", 10000, pms)
        self.epsilon    = None

        # Standard EI class call
        self.ei_analytical = EI(spaces, model)

    def set_best(self, xb: ndarray, yb: float) -> None:
        """
        Sets the current best observation

        Args:
            xb: The best point found so far
            yb: The function value at the best point
        """
        self.xb = xb
        self.yb = yb
        self.ei_analytical.set_best(xb, yb)

    def qei(self, x: ndarray) -> float:
        """
        Computes the q-EI at a set of points using Monte Carlo

        Args:
            x: A NumPy array of points, shape (1, q*d) or (q*d,)

        Returns:
            The q-EI value
        """
        x_flat     = x.flatten()
        q          = len(x_flat) // self.spaces.dim
        x_reshaped = x_flat.reshape((q, self.spaces.dim))

        # Normalize
        x_norm = self.model.normalize(x_reshaped,
                                      self.model.spaces.xmin,
                                      self.model.spaces.xmax)

        # Predictive mean and covariance
        c  = self.model.kernel(x_norm, self.model.x_)
        mu = np.matmul(c, self.model.solve_linsys(self.model.L_, self.model.y_))

        s   = np.matmul(c, self.model.solve_linsys(self.model.L_, c.T))
        cov = self.model.kernel(x_norm, x_norm) - s

        # Add jitter for numerical stability during Cholesky
        cov   += np.eye(q) * 1e-8
        L_pred = cholesky(cov)

        # Fix random samples for deterministic finite differences
        self.epsilon = np.random.randn(self.mc_samples, q)
        # if self.epsilon is None or self.epsilon.shape != (self.mc_samples, q):
        #     rng = np.random.RandomState(42) # Fixed seed


        # Reparameterization: Y = mu + epsilon @ L_pred.T
        Y = mu + np.matmul(self.epsilon, L_pred.T)

        # Improvement: since we minimize, improvement is max(0, yb - min(Y_i))
        min_Y = np.min(Y, axis=1) # min over the q points for each MC sample
        improvement = np.maximum(0.0, self.yb - min_Y)

        return np.mean(improvement)

    def __call__(self, x: ndarray) -> ndarray:
        """
        Computes the q-EI at a set of points

        Args:
            x: A NumPy array of points

        Returns:
            A 1D NumPy array with the scalar q-EI value
        """
        if x.ndim == 1:
            return np.array([self.qei(x)])
        else:
            return self.ei_analytical(x)
