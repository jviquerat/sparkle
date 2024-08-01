# Generic imports
import numpy as np
from   numpy import matmul
from   numpy.linalg import solve

from .ga import genetic_algorithm

# Custom imports
from sparkle.src.utils.error  import error
from sparkle.src.utils.prints import spacer

###############################################
### Kriging model
class kriging():
    def __init__(self):

        self.reset()

    # Reset model
    def reset(self):

        self.theta_ = None
        self.K_     = None
        self.L_     = None
        self.x_     = None
        self.y_     = None
        self.ns_    = None
        self.nf_    = None

    # Build model from input
    def build(self, x, y):

        self.reset()

        self.x_ = x
        self.y_ = y
        self.ns_ = x.shape[0] # nb of samples
        self.nf_ = x.shape[1] # nb of features

        #self.theta_ = np.ones(self.nf_)*0.1
        if (self.theta_ is None):
            bounds = [[-2.0, 1.0]*self.nf_, [-3.0, 1.0], [-3.0, 1.0]]
            theta, score = genetic_algorithm(self.log_likelihood,
                                             bounds, 32, 10, 100, 0.9,
                                             1.0/(32*len(bounds)))
            self.theta_ = np.exp(theta)

        self.K_ = self.kernel(self.x_, self.x_, self.theta_)
        self.L_ = np.linalg.cholesky(self.K_)

    # Evaluate at test points
    def evaluate(self, xt, theta=None):

        if theta is None: theta = self.theta_

        Lt  = np.linalg.solve(self.L_, self.kernel(self.x_, xt, theta))
        mu  = np.dot(Lt.T, np.linalg.solve(self.L_, self.y_))
        Kt  = self.kernel(xt, xt, theta)
        std = np.sqrt(np.diag(Kt) - np.sum(Lt**2, axis=0))

        return mu, std

    # Compute log-likelihood
    def log_likelihood(self, log_theta):

        theta = np.exp(log_theta)

        self.K_ = self.kernel(self.x_, self.x_, theta)
        self.L_ = np.linalg.cholesky(self.K_)

        ones    = np.ones(self.ns_)
        mu      = matmul(ones.T, solve(self.K_, self.y_))
        mu     /= matmul(ones.T, solve(self.K_, ones))
        res     = self.y_ - mu
        var     = matmul(res.T, solve(self.K_, res))/self.ns_
        if (var < 0.0): return 1.0e10
        log_lkh =-0.5*(-self.ns_*np.log(var) + np.linalg.slogdet(self.K_)[1])

        return log_lkh

    def kernel(self, x, y, theta):

        nx  = x.shape[0] # nb of samples in x
        ny  = y.shape[0] # nx of samples in y
        corr = np.zeros((nx,ny))

        for i in range(nx):
            for j in range(ny):
                corr[i,j] = self.cov_function(theta, abs(x[i] - y[j]))
                if (i==j): corr[i,j] += 1.0e-8

        return corr

    # Compute covariance function
    def cov_function(self, theta, dist):

        val = 1.0

        for i in range(dist.shape[0]):
            val *= np.exp(-0.5*np.square(dist[i]/theta[i]))

        return theta[-2]*val + theta[-1]
