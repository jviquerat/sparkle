# Generic imports
import random
import numpy as np

# Custom imports
from sparkle.src.utils.prints import *

###############################################
### Particle swarm optimization
class pso():
    def __init__(self, path, dim, xmin, xmax, pms):

        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.n_steps_max = 20
        self.n_particles = 20
        self.v0          = 0.1
        self.c1          = 0.5
        self.c2          = 0.5
        self.w           = 0.8

        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        if hasattr(pms, "n_particles"): self.n_particles = pms.n_particles
        if hasattr(pms, "v0"):          self.v0          = pms.v0
        if hasattr(pms, "c1"):          self.c1          = pms.c1
        if hasattr(pms, "c2"):          self.c2          = pms.c2
        if hasattr(pms, "w"):           self.w           = pms.w

        self.n_steps_total = self.n_steps_max*self.n_particles

        self.summary()

    # Print informations
    def summary(self):

        spacer()
        print("Using PSO algorithm with "+str(self.n_particles)+" points")
        spacer()
        print("Problem dimensionality is "+str(self.dim))

    # Reset
    def reset(self, run):

        # Step counter       (one step = n_particles cost evaluations)
        # Total step counter (one total step = 1 particle cost evaluation)
        self.stp       = 0
        self.total_stp = 0

        # Path
        self.path = self.base_path+"/"+str(run)

        # Data storage
        self.hist_t = np.zeros((self.n_steps_total))           # time
        self.hist_c = np.zeros((self.n_steps_total))           # cost
        self.hist_b = np.zeros((self.n_steps_total))           # best cost
        self.hist_x = np.zeros((self.n_steps_total, self.dim)) # dofs

        # Positions and velocities
        self.x = np.random.rand(self.n_particles, self.dim)
        self.x = self.xmin + self.x*(self.xmax-self.xmin)
        self.v = np.random.randn(self.n_particles, self.dim)*self.v0

        # Local best and global best
        self.p_best  = np.copy(self.x)
        self.p_score = np.ones(self.n_particles)*1.0e8
        self.g_best  = np.zeros(self.dim)
        self.g_score = 1.0e8

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

