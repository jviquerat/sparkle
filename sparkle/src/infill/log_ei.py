# Generic imports
import numpy as np
from math import sqrt, pi, exp, erf, log, expm1, log1p

###############################################
### Log expected improvement infill
class LogEI():
    def __init__(self, spaces, model):

        self.spaces = spaces
        self.model  = model
        self.bound  =-1.0e8

    def set_best(self, xb, yb):

        self.xb = xb
        self.yb = yb

    def _log_ei(self, x):

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

    # () operator used for optimization
    def __call__(self, x):

        return self._log_ei(x)

# Numerically stable version of log(1 - exp(x))
def log1mexp(x):

    if (x > -log(2)):
        return log(-expm1(x))
    else:
        return log1p(-exp(x))

# erfcx approximation from:
# "Closed-form approximations to the error and complementary error
#  functions and their applications in atmospheric science", Ren et al (2007)
def erfcx(x):

    a = 2.9110
    v = a/((a-1.0)*sqrt(pi*x*x) + sqrt(pi*x*x + a*a))

    return v
