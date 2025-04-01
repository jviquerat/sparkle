# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.agent.lbfgsb import LBFGSB
from sparkle.src.pex.mlhs     import MLHS
from sparkle.src.env.spaces   import EnvSpaces

###############################################
### Multi-start L-BFGS-B
class MSLBFGSB():

    def __init__(self):
        pass

    def optimize(self, f, xmin, xmax, df=None, n_pts=10, m=5, tol=1e-3, max_iter=20):

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
