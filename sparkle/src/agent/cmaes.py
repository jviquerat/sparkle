# Generic imports
import math
import random
import numpy as np

###############################################
### CMAES
class cmaes():
    def __init__(self, path, dim, xmin, xmax, pms):

        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.n_steps_max = 20
        self.sigma0      = 0.25*(xmax-xmin)
        self.x0          = np.zeros(self.dim)
        self.lmbda       = 4 + math.floor(3.0*math.log(self.dim))

        if hasattr(pms, "n_steps_max"):  self.n_steps_max  = pms.n_steps_max
        if hasattr(pms, "lambda"):       self.lmbda        = pms.lmbda
        if hasattr(pms, "sigma0"):       self.sigma0       = pms.sigma0
        if hasattr(pms, "x0"):           self.x0           = np.array(pms.x0)

        # Number of selected samples
        self.fmu    = self.lmbda/2.0
        self.mu     = math.floor(self.fmu)

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

        # Data storage
        self.n_steps_total = self.n_steps_max*self.lmbda
        self.hist_t        = np.zeros((self.n_steps_total))           # time
        self.hist_c        = np.zeros((self.n_steps_total))           # cost
        self.hist_b        = np.zeros((self.n_steps_total))           # best cost
        self.hist_x        = np.zeros((self.n_steps_total, self.dim)) # dofs

    # Reset
    def reset(self, run):

        # Step counter       (one step = lambda cost evaluations)
        # Total step counter (one total step = 1 offspring cost evaluation)
        self.stp = 0
        self.total_stp = 0

        # Best values
        self.best_score    = 1.0e8
        self.best_x        = np.zeros(self.dim)

        # Path
        self.path = self.base_path+"/"+str(run)

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

        # Initial sampling
        # This fills x and z arrays with samples
        self.sample()

        return self.x

    # Sample from distribution
    def sample(self):

        self.z = np.random.randn(self.lmbda, self.dim) # draw from N(0,1)
        self.x = np.zeros_like(self.z)
        for i in range(self.lmbda):
            self.x[i,:] = self.xm[:] + self.sigma*np.matmul(self.BD, self.z[i,:])

    # Step
    def step(self, c):

        # Sort
        self.sort(c)

        # Update best value
        if (c[0] < self.best_score):
            self.best_score = c[0]
            self.best_x     = self.x[0,:]

        # Store
        self.store(c)

        # Update xmean and zmean
        self.xm[:] = 0.0
        self.zm[:] = 0.0
        for i in range(self.mu):
            self.xm[:] += self.x[i,:]*self.w[i]
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

        # Sample
        self.sample()

        self.stp += 1

    # Sort offsprings based on cost
    # x and c arrays are actually modified here
    def sort(self, c):

        sc        = np.argsort(c)
        self.x[:] = self.x[sc[:]]
        self.z[:] = self.z[sc[:]]
        c[:]      = c[sc[:]]

    # Return degrees of freedom
    def dof(self):

        return self.x

    # Return number of degress of freedom
    def ndof(self):

        return self.lmbda

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max):
            return True

        return False

    # Store data
    def store(self, c):

        for i in range(self.lmbda):
            self.hist_t[self.total_stp]   = self.total_stp
            self.hist_x[self.total_stp,:] = self.x[i,:]
            self.hist_c[self.total_stp]   = c[i]
            self.hist_b[self.total_stp]   = self.best_score

            self.total_stp += 1

    # Dump data
    def dump(self):

        filename = self.path+'/raw.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(self.hist_t, (-1,1)),
                              np.reshape(self.hist_c, (-1,1)),
                              np.reshape(self.hist_b, (-1,1)),
                              np.reshape(self.hist_x, (-1,self.dim))]),
                   fmt='%.5e')

    # Print
    def print(self):

        # Total nb of evaluations
        n_eval = self.stp*self.lmbda

        # Handle no-printing after max step
        if (self.stp < self.n_steps_max-1):
            end = "\r"
            self.cnt = 0
        else:
            end  = "\n"
            self.cnt += 1

        # Actual print
        if (self.cnt <= 1):
            gs = f"{self.best_score:.5e}"
            gb = np.array2string(self.best_x, precision=5,
                                 threshold=5, separator=',')
            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" at x = "+str(gb)+"                 ", end=end)

