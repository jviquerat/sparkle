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
        self.sigma       = 0.5
        self.lambda      = 4 + math.floor(3.0*math.log(self.dim))

        if hasattr(pms, "n_steps_max"):  self.n_steps_max  = pms.n_steps_max
        if hasattr(pms, "lambda"):       self.lambda       = pms.lambda
        if hasattr(pms, "sigma"):        self.sigma        = pms.sigma

        self.mu     = math.floor(self.lambda/2)                               # nb of selected offsprings
        self.w      = -np.log(np.arange(1,self.mu)) + math.log(self.mu + 0.5) # recombination weights
        self.w      = self.w/np.sum(self.w)                                   # normalize weights
        self.mu_eff = (np.sum(self.w))**2/(np.sum(np.square(self.w)))         # effective sample size

        # shortcuts for following expressions
        dim   = self.dim
        mueff = self.mu_eff

        self.cc = (4.0 + mueff/dim)/(dim + 4.0 + 2.0*mu_eff/dim) # constant for C evolution path
        self.cs = (mu_eff + 2.0)/(dim + mu_eff + 5.0)            # constant for step size evolution path
        self.c1 = 2.0/((dim + 1.3)**2 + mu_eff)                  # constant for rank-one evolution path
        self.cm = min(1.0 - self.c1,
                      2.0*(mu_eff - 2.0 + 1.0/mu_eff)/((dim+2.0)**2 + mueff))          # constant for rank-mu update
        self.dp = 1.0 + 2.0*max(0.0, math.sqrt((mueff-1.0)/(dim+1.0)) - 1.0) + self.c1 # damping for step-size
        self.cn = math.sqrt(dim)*(1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))             # expectation of N(0,I)

        # Data storage
        self.n_steps_total = self.n_steps_max*self.n_particles
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

        # Path
        self.path = self.base_path+"/"+str(run)

        # Arrays
        self.pc = np.zeros(self.dim)    # C evolution path
        self.ps = np.zeros(self.dim)    # sigma evolution path
        self.B  = np.identity(self.dim) # coordinate system
        self.D  = np.identity(self.dim) # scaling matrix
        self.C  = np.identity(self.dim) # covariance matrix

        return self.x

    # Step
    # Data storage is performed between update of best points
    # and update of positions and velocities so the recorded (x,v)
    # matches with the correct cost
    def step(self, c):

        self.update_best(c)
        self.store(c)
        self.update_xv()

        self.stp += 1

    # Update local and global best
    def update_best(self, c):

        for i in range(self.n_particles):

            # Update best local score
            if (c[i] <= self.p_score[i]):
                self.p_score[i]  = c[i]
                self.p_best[i,:] = self.x[i,:]

            # Update best global score
            if (c[i] <= self.g_score):
                self.g_score   = c[i]
                self.g_best[:] = self.x[i,:]

    # Update positions and velocities
    def update_xv(self):

        for i in range(self.n_particles):
            r1, r2 = np.random.rand(2)
            self.v[i,:]  = (self.w*self.v[i,:] +
                            self.c1*r1*(self.p_best[i,:] - self.x[i,:]) +
                            self.c2*r2*(self.g_best[:]   - self.x[i,:]))
            v = np.random.randn(self.n_particles, self.dim)*self.v0
            self.x[i,:] += self.v[i,:]

    # Return degrees of freedom
    def dof(self):

        return self.x

    # Return number of degress of freedom
    def ndof(self):

        return self.n_particles

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max):
            return True

        return False

    # Store data
    def store(self, c):

        for i in range(self.n_particles):
            self.hist_t[self.total_stp]   = self.total_stp
            self.hist_x[self.total_stp,:] = self.x[i,:]
            self.hist_c[self.total_stp]   = c[i]
            self.hist_b[self.total_stp]   = self.g_score

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
        n_eval = self.stp*self.n_particles

        # Handle no-printing after max step
        if (self.stp < self.n_steps_max-1):
            end = "\r"
            self.cnt = 0
        else:
            end  = "\n"
            self.cnt += 1

        # Actual print
        if (self.cnt <= 1):
            gs = f"{self.g_score:.5e}"
            gb = np.array2string(self.g_best, precision=5,
                                 threshold=5, separator=',')
            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" at x = "+str(gb)+"                 ", end=end)

