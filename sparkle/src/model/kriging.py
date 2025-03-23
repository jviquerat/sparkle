# Generic imports
import numpy as np
from   numpy import matmul
from   numpy.linalg import solve

# Custom imports
from sparkle.src.model.base      import base_model
from sparkle.src.kernel.kernel   import kernel_factory
from sparkle.src.utils.default   import set_default
from sparkle.src.utils.error     import error

###############################################
### Kriging model
class kriging(base_model):
    def __init__(self, spaces, path, pms):
        super().__init__(spaces, path)

        # Set parameters
        self.optimize_every_ = set_default("optimize_every", 1000, pms)
        self.load_model_     = set_default("load_model", False, pms)

        # Initialize kernel
        self.kernel = kernel_factory.create(pms.kernel.name,
                                            spaces = spaces,
                                            pms    = pms.kernel)

        # Check model loading
        if (self.load_model_):
            if not hasattr(pms, "model_file"):
                error("kriging", "__init__",
                      "load_model option requires model_file parameter")
            self.model_file_ = pms.model_file

        self.reset()

    # Reset model
    def reset(self):

        self.K_  = None
        self.x_  = None
        self.y_  = None
        self.it  = 0

        self.kernel.reset()

    # Build model from input
    def build(self, x, y):

        self.x_ = self.normalize(x)
        self.y_ = y

        if (self.it%self.optimize_every_ == 0):
            self.kernel.optimize(self.x_, self.y_)

        self.it += 1
        self.K_  = self.kernel(self.x_, self.x_)

    # Evaluate at test points
    def evaluate(self, xt):

        xn  = self.normalize(xt)
        Kl  = self.kernel(xn, self.x_)
        mu  = matmul(Kl, solve(self.K_, self.y_))
        Kt  = self.kernel(xn, xn)
        std = np.diag(Kt - matmul(Kl, solve(self.K_, Kl.T)))
        std = np.sqrt(np.abs(std))

        return mu, std

    # Dump kriging data
    def dump(self, filename="kriging.dat"):

        filename = self.path+"/"+filename
        with open(filename, "w") as f:
            f.write(f"{self.kernel.nf_} \n")
            f.write(f"{self.kernel.ns_} \n")
            f.write(f"{self.kernel.dim_} \n")
            f.write(f"{self.kernel.diag_eps_} \n")
            np.savetxt(f, self.kernel.theta_)
            np.savetxt(f, self.K_)
            np.savetxt(f, self.x_)
            np.savetxt(f, self.y_)

    # Load kriging data
    def load(self, filename=None):

        self.reset()
        self.it += 1

        if filename is None: filename = self.model_file_

        with open(filename, "r") as f:
            self.kernel.nf_       = int(f.readline().split(" ")[0])
            self.kernel.ns_       = int(f.readline().split(" ")[0])
            self.kernel.dim_      = int(f.readline().split(" ")[0])
            self.kernel.diag_eps_ = float(f.readline().split(" ")[0])

            self.K_     = np.zeros((self.kernel.ns_, self.kernel.ns_))
            self.x_     = np.zeros((self.kernel.ns_, self.kernel.nf_))
            self.y_     = np.zeros(self.kernel.ns_)

            l = 4
            self.kernel.theta_ = np.loadtxt(filename,
                                            skiprows=l,
                                            max_rows=self.kernel.dim_,
                                            ndmin=1)

            l += self.kernel.dim_
            for i in range(self.kernel.dim_): f.readline()
            self.K_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.kernel.ns_)

            l += self.kernel.ns_
            for i in range(self.kernel.ns_): f.readline()
            self.x_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.kernel.ns_)

            l += self.kernel.ns_
            for i in range(self.kernel.ns_): f.readline()
            self.y_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.kernel.ns_)
