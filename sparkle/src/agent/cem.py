# Generic imports
import math
import random
import numpy as np

###############################################
### CEM
class cem():
    def __init__(self, path, dim, xmin, xmax, pms):

        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.n_steps_max = 20
        self.n_points    = 2*self.dim
        self.n_elites    = math.floor(self.n_points/2)
        self.alpha       = 0.2

        if hasattr(pms, "n_steps_max"):  self.n_steps_max  = pms.n_steps_max
        if hasattr(pms, "n_points"):     self.n_points     = pms.n_points
        if hasattr(pms, "n_elites"):     self.n_elites     = pms.n_elites
        if hasattr(pms, "alpha"):        self.alpha        = pms.alpha

        # Data storage
        self.n_steps_total = self.n_steps_max*self.n_points
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
        self.best_score = 1.0e8
        self.best_x     = np.zeros(self.dim)

        # Path
        self.path = self.base_path+"/"+str(run)

        # Min and max arrays used for cem adaptation
        self.xmin_cem = self.xmin.copy()
        self.xmax_cem = self.xmax.copy()

        # Initial sampling
        # This fills x and z arrays with samples
        self.sample()

        return self.x

    # Sample from distribution
    def sample(self):

        self.x = np.zeros((self.n_points, self.dim))

        for i in range(self.n_points):
            self.x[i,:] = np.random.uniform(low  = self.xmin_cem,
                                            high = self.xmax_cem)

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

        # Update xmin and xmax
        xmin = np.amin(self.x[:self.n_elites,:], axis=0)
        xmax = np.amax(self.x[:self.n_elites,:], axis=0)
        self.xmin_cem[:] = ((1.0-self.alpha)*self.xmin_cem[:] +
                            self.alpha*xmin[:])
        self.xmax_cem[:] = ((1.0-self.alpha)*self.xmax_cem[:] +
                            self.alpha*xmax[:])

        # Sample
        self.sample()

        self.stp += 1

    # Sort offsprings based on cost
    # x and c arrays are actually modified here
    def sort(self, c):

        sc        = np.argsort(c)
        self.x[:] = self.x[sc[:]]
        c[:]      = c[sc[:]]

    # Return degrees of freedom
    def dof(self):

        return self.x

    # Return number of degress of freedom
    def ndof(self):

        return self.n_points

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max):
            return True

        return False

    # Store data
    def store(self, c):

        for i in range(self.n_points):
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
        n_eval = self.stp*self.n_points

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
            gb = np.array2string(self.best_x, precision=5, floatmode='fixed',
                                 threshold=5, separator=',')
            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" at x = "+str(gb)+"                                                                                   ", end=end)

