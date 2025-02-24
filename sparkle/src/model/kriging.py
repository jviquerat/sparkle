# Generic imports
import numpy as np
from   numpy import matmul
from   numpy.linalg import solve

# Custom imports
from sparkle.src.agent.optimizer import optimizer
from sparkle.src.env.spaces      import environment_spaces
from sparkle.src.utils.default   import set_default
from sparkle.src.utils.error     import error
from sparkle.src.utils.prints    import spacer

###############################################
### Kriging model
class kriging():
    def __init__(self, spaces, pms):

        self.spaces           = spaces
        self.recompute_theta_ = set_default("recompute_theta", False, pms)
        self.load_model_      = set_default("load_model", False, pms)

        if (self.load_model_):
            if not hasattr(pms, "model_file"):
                error("ego", "__init__",
                      "load_model option requires model_file parameter")
            self.model_file_ = pms.model_file

        self.reset()

    # Reset model
    def reset(self):

        self.diag_eps_ = 1.0e-8

        self.K_  = None
        self.x_  = None
        self.y_  = None
        self.ns_ = None
        self.nf_ = None

        if (self.recompute_theta_): self.theta_ = None

    # Normalize inputs
    def normalize(self, x):

        xx = (x - self.spaces.xmin)/(self.spaces.xmax - self.spaces.xmin)
        return xx

    # Denormalize inputs
    def denormalize(self, x):

        xx = self.spaces.xmin + (self.spaces.xmax - self.spaces.xmin)*x
        return xx

    @property
    def x(self):
        return self.denormalize(self.x_)

    @property
    def y(self):
        return self.y_

    # Build model from input
    def build(self, x, y):

        self.reset()

        self.x_   = self.normalize(x)
        self.y_   = y
        self.ns_  = x.shape[0] # nb of samples
        self.nf_  = x.shape[1] # nb of features
        self.dim_ = self.nf_ + 2

        if (self.recompute_theta_ or not hasattr(self, "theta_")):
            name        = "cmaes"
            x0          = np.zeros(self.dim_)
            xmin        =-2.0*np.ones(self.dim_)
            xmin[-2:]   =-3.0
            xmax        = np.ones(self.dim_)
            n_points    = 200
            n_steps_max = 10

            loc_space = {"dim": self.dim_, "x0": x0, "xmin": xmin, "xmax": xmax}
            s   = environment_spaces(loc_space)
            opt = optimizer(name, s, n_points, n_steps_max, self.log_likelihood)
            theta, c = opt.optimize()

            self.theta_ = np.exp(theta)

        self.K_ = self.kernel(self.x_, self.x_, self.theta_)

    # Evaluate at test points
    def evaluate(self, xt, theta=None):

        if theta is None: theta = self.theta_

        xn  = self.normalize(xt)
        Kl  = self.kernel(xn, self.x_, theta)
        mu  = matmul(Kl, solve(self.K_, self.y_))
        Kt  = self.kernel(xn, xn, theta)
        std = np.diag(Kt - matmul(Kl, solve(self.K_, Kl.T)))
        std = np.sqrt(np.abs(std))

        return mu, std

    # Compute log-likelihood
    def log_likelihood(self, log_theta):

        theta = np.exp(log_theta)

        K_   = self.kernel(self.x_, self.x_, theta)
        ones = np.ones(self.ns_)
        mu   = matmul(ones.T, solve(K_, self.y_))
        mu  /= matmul(ones.T, solve(K_, ones))

        res     = self.y_ - mu
        var     = matmul(res.T, solve(K_, res))/self.ns_
        if (var < 0.0): return 1.0e10
        log_lkh =-0.5*(-self.ns_*np.log(var) + np.linalg.slogdet(K_)[1])

        return log_lkh

    def kernel(self, x, y, theta):

        nx  = x.shape[0] # nb of samples in x
        ny  = y.shape[0] # nx of samples in y
        corr = np.zeros((nx,ny))

        for i in range(nx):
            for j in range(ny):
                corr[i,j] = self.cov_function(theta, abs(x[i] - y[j]))
                if (i==j): corr[i,j] += self.diag_eps_

        return corr

    # Compute covariance function
    def cov_function(self, theta, dist):

        val = 1.0

        for i in range(dist.shape[0]):
            val *= np.exp(-0.5*np.square(dist[i]/theta[i]))

        return theta[-2]*val + theta[-1]

    # Dump kriging data
    def dump(self, filename):

        with open(filename, "w") as f:
            f.write(f"{self.nf_} \n")
            f.write(f"{self.ns_} \n")
            f.write(f"{self.dim_} \n")
            f.write(f"{self.diag_eps_} \n")
            np.savetxt(f, self.theta_)
            np.savetxt(f, self.K_)
            np.savetxt(f, self.x_)
            np.savetxt(f, self.y_)

    # Load kriging data
    def load(self, filename=None):

        self.reset()

        if filename is None: filename = self.model_file_

        with open(filename, "r") as f:
            self.nf_       = int(f.readline().split(" ")[0])
            self.ns_       = int(f.readline().split(" ")[0])
            self.dim_      = int(f.readline().split(" ")[0])
            self.diag_eps_ = float(f.readline().split(" ")[0])

            self.theta_ = np.zeros(self.dim_)
            self.K_     = np.zeros((self.ns_, self.ns_))
            self.x_     = np.zeros((self.ns_, self.nf_))
            self.y_     = np.zeros(self.ns_)

            l = 4
            self.theta_ = np.loadtxt(filename,
                                     skiprows=l,
                                     max_rows=self.dim_)

            l += self.dim_
            for i in range(self.dim_): f.readline()
            self.K_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.ns_)

            l += self.ns_
            for i in range(self.ns_): f.readline()
            self.x_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.ns_)

            l += self.ns_
            for i in range(self.ns_): f.readline()
            self.y_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.ns_)
