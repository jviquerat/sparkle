import math
from types import SimpleNamespace

import numpy as np
from numpy import ndarray

from sparkle.src.agent.base import BaseAgent
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default


###############################################
### CMAES
class CMAES(BaseAgent):
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: SimpleNamespace) -> None:
        super().__init__(path, spaces, pms)

        self.name        = "CMAES"
        sg0              = 0.25*(np.min(self.xmax)-np.max(self.xmin))
        self.sigma0      = set_default("sigma0", sg0, pms)
        npts             = 4 + math.floor(3.0*math.log(self.dim))
        self.n_points    = set_default("n_points", npts, pms)
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.clip        = set_default("clip", False, pms)
        self.escape      = set_default("escape", False, pms)

        # Number of selected samples
        self.fmu = self.n_points/2.0
        self.mu  = math.floor(self.fmu)

        # Recombination weights
        self.w = np.zeros(self.mu)
        for i in range(self.mu):
            self.w[i] = math.log(self.fmu + 0.5) - math.log(i+1.0)
        self.w = self.w/np.sum(self.w)

        # Effective sample size
        self.mu_eff = (np.sum(self.w))**2/(np.sum(np.square(self.w)))

        # Shortcuts for following expressions
        dim    = float(self.dim)
        mu_eff = float(self.mu_eff)

        self.cc = (4.0 + mu_eff/dim)/(dim + 4.0 + 2.0*mu_eff/dim) # constant for C evolution path
        self.cs = (mu_eff + 2.0)/(dim + mu_eff + 5.0)             # constant for step size evolution path
        self.c1 = 2.0/((dim + 1.3)**2 + mu_eff)                   # constant for rank-one evolution path
        self.cmu = min(1.0 - self.c1,
                       2.0*(mu_eff - 2.0 + 1.0/mu_eff + 0.25)/((dim+2.0)**2 + mu_eff))  # constant for rank-mu update
        self.dp = 1.0 + 2.0*max(0.0, math.sqrt((mu_eff-1.0)/(dim+1.0)) - 1.0) + self.cs # damping for step-size
        self.cn = math.sqrt(dim)*(1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))              # expectation of N(0,I)

        if (not self.silent): self.summary()

    # Reset
    def reset(self, run: int) -> None:

        # Mother class reset
        super().reset(run)

        # Arrays
        self.pc    = np.zeros(self.dim)        # C evolution path
        self.ps    = np.zeros(self.dim)        # sigma evolution path
        self.B     = np.identity(self.dim)     # coordinate system
        self.D     = np.identity(self.dim)     # scaling matrix
        self.BD    = np.matmul(self.B, self.D) # for efficiency
        self.C     = np.identity(self.dim)     # covariance matrix
        self.xm    = self.x0                   # mean vector
        self.zm    = np.zeros(self.dim)        # auxiliary mean vector
        self.sigma = self.sigma0               # global standard deviation

    # Sample from distribution
    def sample(self) -> ndarray:

        x      = np.zeros((self.n_points, self.dim))
        self.z = np.random.randn(self.n_points, self.dim) # draw from N(0,1)
        for i in range(self.n_points):
            x[i,:] = self.xm[:] + self.sigma*np.matmul(self.BD, self.z[i,:])

        if (self.clip):
            for i in range(self.dim):
                x[:,i] = np.clip(x[:,i], self.xmin[i], self.xmax[i])

        return x

    # Step
    def step(self, x: ndarray, c: ndarray) -> None:

        # Sort
        self.sort(x, c)

        # Update xmean and zmean
        self.xm[:] = 0.0
        self.zm[:] = 0.0
        for i in range(self.mu):
            self.xm[:] += x[i,:]*self.w[i]
            self.zm[:] += self.z[i,:]*self.w[i] # = D^-1 * B^T * (x_mean-x_old)/sigma

        # Update ps
        coeff   = math.sqrt(self.cs*(2.0-self.cs)*self.mu_eff)
        self.ps = (1.0-self.cs)*self.ps + coeff*np.matmul(self.B, self.zm)

        # Update pc
        coeff   = math.sqrt(self.cc*(2.0-self.cc)*self.mu_eff)
        norm_ps = np.linalg.norm(self.ps)
        hs      = float(norm_ps/math.sqrt(1.0 - (1.0-self.cs)**(2.0*(self.stp+1))) < (1.4 + 2.0/(self.stp+1))*self.cn)
        self.pc = (1.0-self.cc)*self.pc + hs*coeff*np.matmul(self.BD, self.zm)

        # Update C
        y  = np.zeros((self.mu, self.dim))
        for i in range(self.mu):
            y [i,:] = np.matmul(self.BD, self.z[i,:])

        self.C = ((1.0-self.c1-self.cmu)*self.C +
                  self.c1*(np.outer(self.pc,self.pc) + (1.0-hs)*self.cc*(2.0-self.cc)*self.C) +
                  self.cmu*(np.matmul(np.transpose(y),np.matmul(np.diag(self.w), y))))

        # Update sigma
        self.sigma = self.sigma*np.exp(min(1.0,(self.cs/self.dp)*(norm_ps/self.cn - 1.0)))

        # Update B and D
        self.C = np.triu(self.C) + np.transpose(np.triu(self.C,1))
        self.D, self.B = np.linalg.eigh(self.C)
        self.D         = np.diag(np.sqrt(self.D))
        self.BD        = np.matmul(self.B, self.D)

        # Escape flat cost function
        if (self.escape):
            if (c[0] > 0.999*c[-1]):
                self.sigma *= np.exp(2.0+self.cs/self.dp)

        self.stp += 1

    # Sort offsprings based on cost
    # x and c arrays are actually modified here
    def sort(self, x: ndarray, c: ndarray) -> None:

        sc        = np.argsort(c)
        x[:]      = x[sc[:]]
        self.z[:] = self.z[sc[:]]
        c[:]      = c[sc[:]]
