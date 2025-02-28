# Generic imports
import numpy as np
from math import sqrt, pi, exp, erf

# Custom imports
#from sparkle.src.kernel.base   import base_kernel
#from sparkle.src.utils.default import set_default

###############################################
### Expected improvement infill
class ei():
    def __init__(self, spaces, model):

        self.spaces = spaces
        self.model  = model

    def set_best(self, xb, yb):

        self.xb = xb
        self.yb = yb

    # () operator
    def __call__(self, x):

        x       = np.reshape(x, (-1,self.spaces.dim))
        mu, std = self.model.evaluate(x)

        n  = x.shape[0]
        ei = np.zeros(n)
        for i in range(n):
            prob      = (self.yb - mu[i])/std[i]
            cum_dist  = 0.5*(1.0 + erf(prob/sqrt(2.0)))
            prob_dist = (1.0/sqrt(2.0*pi))*np.exp(-0.5*prob**2)
            ei[i]     = std[i]*(prob*cum_dist + prob_dist)

        return -ei
