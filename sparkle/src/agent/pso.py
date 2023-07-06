# Generic imports
import random
import numpy as np

# Custom imports

###############################################
### Particle swarm optimization
class pso():
    def __init__(self, dim, pms):

        self.dim         = dim
        self.n_steps_max = 20
        self.n_particles = 20
        self.v0          = 0.1

        self.c1 = 0.1
        self.c2 = 0.1
        self.w  = 0.8

        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        if hasattr(pms, "n_particles"): self.n_particles = pms.n_particles
        if hasattr(pms, "v0"):          self.v0          = pms.v0

        self.reset()

    # Reset
    def reset(self):

        # Iteration counter
        self.stp = 0

        # Positions and velocities
        self.x = np.random.rand(self.n_particles, self.dim)
        self.x = 2.0*self.x - 1.0 # Scale to [-1,1]
        self.v = np.random.randn(self.n_particles, self.dim)*self.v0

        # Local best and global best
        self.p_best  = np.copy(self.x)
        self.p_score = np.ones(self.n_particles)*1.0e8
        self.g_best  = np.zeros((2))
        self.g_score = 1.0e8

        return self.x

    # Step
    def step(self, c):

        self.update_best(c)
        self.update_xv()

        self.stp += 1

    # Update local and global best
    def update_best(self, c):

        for i in range(self.n_particles):

            # Update best local score
            if (c[i] >= self.p_score[i]):
                self.p_score[i]  = c[i]
                self.p_best[i,:] = self.x[i,:]

            # Update best global score
            if (c[i] >= self.g_score):
                self.g_score   = c[i]
                self.g_best[:] = self.x[i,:]

    # Update positions and velocities
    def update_xv(self):

        r1, r2 = np.random.rand(2)
        for i in range(self.n_particles):
            self.v[i,:]  = (self.w*self.v[i,:] +
                            self.c1*r1*(self.p_best[i,:] - self.x[i,:]) +
                            self.c2*r2*(self.g_best[:] - self.x[i,:]))
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
