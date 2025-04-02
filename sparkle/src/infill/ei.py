from math import erf, exp, pi, sqrt
from typing import Any

import numpy as np
from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.model.kriging import Kriging


###############################################
### Expected improvement infill
class EI():
    def __init__(self, spaces: EnvSpaces, model: Any) -> None:

        self.spaces = spaces
        self.model  = model

    def set_best(self, xb: ndarray, yb: float) -> None:

        self.xb = xb
        self.yb = yb

    # Actual infill criterion
    def _ei(self, x: ndarray) -> ndarray:

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

    # () operator used for optimization
    def __call__(self, x: ndarray) -> ndarray:

        return self._ei(x)
