import types
from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
import scipy

from sparkle.src.agent.lbfgsb import LBFGSB
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.mlhs import MLHS


###############################################
class MSLBFGSB():
    """
    Multi-start L-BFGS-B optimizer.

    This class implements a multi-start version of the L-BFGS-B algorithm.
    It performs multiple optimizations from different starting points to
    increase the likelihood of finding the global optimum.
    """

    def __init__(self) -> None:
        """
        Initializes the MSLBFGSB optimizer.
        """

    def optimize(self,
                 f: Callable,
                 xmin: ndarray,
                 xmax: ndarray,
                 df: Optional[Callable]=None,
                 n_pts: int=10,
                 m: int=5,
                 tol: float=1e-3,
                 max_iter: int=20,
                 test_ratio: int=10,
                 use_scipy=True) -> Tuple[ndarray, float]:
        """
        Optimizes a function using the multi-start L-BFGS-B algorithm.
        We sample test_ratio*n_pts points and compute the associated cost,
        then select the best n_pts as initial points for the multi-start algorithm

        Args:
            f: The objective function to minimize.
            xmin: The lower bounds for the variables, as a NumPy array.
            xmax: The upper bounds for the variables, as a NumPy array.
            df: An optional function to compute the gradient of f. If not provided,
                finite differences will be used.
            n_pts: The number of starting points for the multi-start optimization.
            m: The maximum number of correction pairs to store in L-BFGS-B (memory size).
            tol: The tolerance for the norm of the projected gradient in L-BFGS-B.
            max_iter: The maximum number of iterations for each L-BFGS-B optimization.
            test_ratio: Nb of points ratio for initial values test
            use_scipy: Whether to use scipy LBFGSB implementation

        Returns:
            A tuple containing:
                - The optimized point (NumPy array).
                - The value of the objective function at the optimized point (float).
        """

        pms          = types.SimpleNamespace()
        pms.n_points = n_pts*test_ratio
        pms.n_iter   = 1000

        dim        = xmin.shape[0]
        space_dict = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
        spaces     = EnvSpaces(space_dict)
        pex        = MLHS(spaces, pms)

        # Compute costs and retain only best n_pts
        costs = np.zeros(n_pts*test_ratio)
        for k in range(n_pts*test_ratio):
            costs[k] = f(pex.x[k]).squeeze()
        best  = np.argsort(costs)[:n_pts]
        x     = pex.x[best]

        x_star = np.zeros((n_pts, dim))
        c_star = np.zeros(n_pts)

        # Use scipy implementation
        if (use_scipy):
            scipy_bounds = list(zip(xmin, xmax))
            scipy_options = {
                'maxcor': m,
                'gtol': tol,
                'maxiter': max_iter,
                'disp': False
                }

            for k in range(n_pts):
                result = scipy.optimize.minimize(
                    fun=f,
                    x0=x[k],
                    method='L-BFGS-B',
                    jac=df,
                    bounds=scipy_bounds,
                    options=scipy_options
                )

                x_star[k] = result.x
                c_star[k] = result.fun

        # Use in-house implementation
        else:
            opt = LBFGSB()

            for k in range(n_pts):
                x, c      = opt.optimize(f, x[k], xmin, xmax,
                                         df=df, m=m, tol=tol, max_iter=max_iter)
                x_star[k] = x
                c_star[k] = c

        # Retrieve best
        best = np.argmin(c_star)

        return x_star[best], c_star[best]
