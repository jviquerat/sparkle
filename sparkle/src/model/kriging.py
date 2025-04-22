from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
from numpy import matmul, ndarray
from numpy.linalg import cholesky, solve

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.kernel.kernel import kernel_factory
from sparkle.src.model.base import BaseModel
from sparkle.src.utils.default import set_default
from sparkle.src.utils.error import error


class Kriging(BaseModel):
    """
    Kriging surrogate model.

    This class implements a Kriging surrogate model, which is a type of
    Gaussian process regression model used for function approximation.
    """
    def __init__(self, spaces: EnvSpaces, path: str, pms: SimpleNamespace) -> None:
        """
        Initializes the Kriging model.

        Args:
            spaces: The environment's search space definition.
            path: The base path for storing results.
            pms: A SimpleNamespace object containing parameters for the model.
        """
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

    def reset(self) -> None:
        """
        Resets the Kriging model.
        """

        self.C_  = None
        self.L_  = None
        self.x_  = None
        self.y_  = None
        self.it  = 0

        self.kernel.reset()

    def build(self, x: ndarray, y: ndarray) -> None:
        """
        Builds the Kriging model from input data.

        Args:
            x: The input data points.
            y: The corresponding target values.
        """

        self.x_ = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        self.y_ = y

        if (self.it%self.optimize_every_ == 0):
            self.kernel.optimize(self.x_, self.y_)

        self.C_  = self.kernel(self.x_, self.x_)
        self.L_  = cholesky(self.C_)

        self.it += 1

    def solve_linsys(self, L: ndarray, b: ndarray) -> ndarray:
        """
        Solves the linear system K * x = b using Cholesky decomposition.
        It first solves L*y = b, then L.T*x = y

        Args:
            L: cholesky matrix
            b: right hand side vector

        Returns:
            x: solution of the linear system
        """

        return solve(L.T, solve(L, b))

    def evaluate(self, x_test: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluates the Kriging model at test points

        mu = c(x) @ C_inv @ y
        std² = diag(c(x,x) - c(x) @ C_inv @ c(x).T)

        Shapes:
            nt is the number of test points
            ns is the number of samples of the model
            d  is the dimension of each point

        Args:
            x_test: The test points at which to evaluate the model, shape (nt, d)

        Returns:
            A tuple containing:
                - The predicted mean values at the test points.
                - The predicted standard deviations at the test points.
        """

        x = self.normalize(x_test, self.spaces.xmin, self.spaces.xmax) # shape (nt, d)

        # Computation of mu
        # C_inv         -> shape (ns, ns)
        # y             -> shape (ns,)
        # C_inv @ y     -> shape (ns)
        # c             -> shape (nt, ns)
        # c @ C_inv @ y -> shape (nt,)
        c  = self.kernel(x, self.x_)
        mu = matmul(c, self.solve_linsys(self.L_, self.y_))

        # Computation of std
        # C_inv           -> shape (ns, ns)
        # c.T             -> shape (ns, nt)
        # C_inv @ c.T     -> shape (ns, nt)
        # c               -> shape (nt, ns)
        # C(x,x)          -> shape (nt, nt)
        # c @ C_inv @ c.T -> shape (nt, nt)
        s = matmul(c, self.solve_linsys(self.L_, c.T))
        std_squared = np.diag(self.kernel(x,x) - s)
        std = np.sqrt(np.abs(std_squared))

        return mu, std

    def evaluate_grad(self, x_test: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluates the gradient of mu and std at test points

        grad_mu = grad_c(x).T @ C_inv @ y
        grad_std = -grad_c(x).T @ C_inv @ c.T / std

        Shapes:
            nt is the number of test points
            ns is the number of samples of the model
            d  is the dimension of each point

        Args:
            x_test: The test points at which to evaluate the gradients of the model, shape (nt, d)

        Returns:
            A tuple containing:
                - The gradient of mu  at the test points, shape (nt, d)
                - The gradient of std at the test points, shape (nt, d)
        """

        # Initial computations
        # x     -> shape (nt, d)
        # c     -> shape (nt, ns)
        # dc_dx -> shape (nt, ns, d)
        x = self.normalize(x_test, self.spaces.xmin, self.spaces.xmax)
        c = self.kernel.covariance(x, self.x_)
        dc_dx = self.kernel.covariance_dx(x, self.x_)

        # Gradient of mu
        # y                          -> shape (ns,)
        # C_inv                      -> shape (ns, ns)
        # C_inv @ y                  -> shape (ns,)
        # dc_dx                      -> shape (nt, ns, d)
        # dc_dx[i,:,:].T             -> shape (d, ns)
        # dc_dx[i,:,:].T @ C_inv @ y -> shape (d,)
        grad_mu = np.zeros_like(x_test)
        for i in range(x_test.shape[0]):
            grad_mu[i, :]  = matmul(dc_dx[i,:,:].T, self.solve_linsys(self.L_, self.y_))
            grad_mu[i, :] /= self.spaces.xmax - self.spaces.xmin

        # Gradient of std
        # C_inv                             -> shape (ns, ns)
        # c                                 -> shape (nt, ns)
        # dc_dx                             -> shape (nt, ns, d)
        # dc_dx[i,:,:].T                    -> shape (d, ns)
        # C_inv @ c[i,:].T                  -> shape (ns, 1)
        # dc_dx[i,:,:].T @ C_inv @ c[i,:].T -> shape (d,)
        grad_std = np.zeros_like(x_test)
        mu, std = self.evaluate(x_test) # Provide the non-normalized x
        for i in range(x_test.shape[0]):
            tmp = matmul(dc_dx[i,:,:].T, self.solve_linsys(self.L_, c[i, :].T))
            grad_std[i, :]  = -tmp/std[i]
            grad_std[i, :] /= self.spaces.xmax - self.spaces.xmin

        return grad_mu, grad_std

    def dump(self, filename: str="kriging.dat") -> None:
        """
        Dumps the Kriging model data to a file.

        Args:
            filename: The name of the file to which to dump the data.
        """

        filename = self.path+"/"+filename
        with open(filename, "w") as f:
            f.write(f"{self.kernel.nf_} \n")
            f.write(f"{self.kernel.ns_} \n")
            f.write(f"{self.kernel.dim_} \n")
            f.write(f"{self.kernel.diag_eps_} \n")
            np.savetxt(f, self.kernel.theta_)
            np.savetxt(f, self.x)
            np.savetxt(f, self.y)

    def load(self, filename: Optional[str]=None) -> None:
        """
        Loads the Kriging model data from a file.

        Args:
            filename: The name of the file from which to load the data.
                      If None, uses the model_file_ attribute.
        """

        self.reset()
        self.it += 1

        if filename is None: filename = self.model_file_

        with open(filename, "r") as f:
            self.kernel.nf_       = int(f.readline().split(" ")[0])
            self.kernel.ns_       = int(f.readline().split(" ")[0])
            self.kernel.dim_      = int(f.readline().split(" ")[0])
            self.kernel.diag_eps_ = float(f.readline().split(" ")[0])

            self.x_     = np.zeros((self.kernel.ns_, self.kernel.nf_))
            self.y_     = np.zeros(self.kernel.ns_)

            l = 4
            self.kernel.theta_ = np.loadtxt(filename,
                                            skiprows=l,
                                            max_rows=self.kernel.dim_,
                                            ndmin=1)

            l += self.kernel.dim_
            for i in range(self.kernel.dim_): f.readline()
            self.x_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.kernel.ns_)

            l += self.kernel.ns_
            for i in range(self.kernel.ns_): f.readline()
            self.y_ = np.loadtxt(filename,
                                 skiprows=l,
                                 max_rows=self.kernel.ns_)

        self.build(self.x_, self.y_)
