# Generic imports
import numpy as np
from math import sqrt, pi, exp, erf

###############################################
### Expected improvement infill
class ei():
    def __init__(self, spaces, model):

        self.spaces = spaces
        self.model  = model

    def set_best(self, xb, yb):

        self.xb = xb
        self.yb = yb

    # Actual infill criterion
    def _ei(self, x):

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

    # Actual infill criterion
    def infill(self, x):

        return self._ei(x)

    # () operator used for optimization
    def __call__(self, x):

        ei = self._ei(x)

        return -ei[0]
