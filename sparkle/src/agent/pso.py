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
        self.step = 0

        # Positions and velocities
        self.x = np.random.rand(self.dim, self.n_particles)
        self.x = 2.0*self.x - 1.0 # Scale to [-1,1]
        self.v = np.random.randn(self.dim, self.n_particles)*self.v0

        # Local best and global best
        self.p_best  = np.copy(self.x.copy)
        self.p_score = np.ones(self.n_particles)*1.0e8
        self.g_best  = np.zeros((2))
        self.g_score = 1.0e8

        return self.x

    # Pre-loop computations
    def pre_loop(self, c):

        self.update_best(c)

    # Update local and global best
    def update_best(self, c):

        for i in range(self.n_particles):

            # Update best local score
            if (c[i] >= self.p_score[i]):
                self.p_score[i] = c[i]
                self.p_best[:,i]  = self.x[:,i]

            # Update best global score
            if (c[i] >= self.g_score):
                self.g_score   = c[i]
                self.g_best[:] = self.x[:,i]

    # Return degrees of freedom
    def dof(self):

        return self.x

    # Check if done
    def done(self):

        if (self.step == self.step_max):
            return True

        return False

    # Perform one optimization step
    def step(self, cost):


        self.step += 1
