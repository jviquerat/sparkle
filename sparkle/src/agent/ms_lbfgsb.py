import types
from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray

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
        pass

    def optimize(self,
                 f: Callable,
                 xmin: ndarray,
                 xmax: ndarray,
                 df: Optional[Callable]=None,
                 n_pts: int=10,
                 m: int=5,
                 tol: float=1e-3,
                 max_iter: int=20) -> Tuple[ndarray, float]:
        """
        Optimizes a function using the multi-start L-BFGS-B algorithm.

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

        Returns:
            A tuple containing:
                - The optimized point (NumPy array).
                - The value of the objective function at the optimized point (float).
        """

        pms          = types.SimpleNamespace()
        pms.n_points = n_pts
        pms.n_iter   = 1000

        dim        = xmin.shape[0]
        space_dict = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
        spaces     = EnvSpaces(space_dict)
        pex        = MLHS(spaces, pms)

        x_star = np.zeros((n_pts, dim))
        c_star = np.zeros(n_pts)
        opt    = LBFGSB()

        for k in range(n_pts):
            x, c      = opt.optimize(f, pex.x[k], xmin, xmax,
                                     df=df, m=m, tol=tol, max_iter=max_iter)
            x_star[k] = x
            c_star[k] = c

        best = np.argmin(c_star)

        return x_star[best], c_star[best]
