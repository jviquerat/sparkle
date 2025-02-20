# Generic imports
import numpy as np

# Custom imports
from sparkle.src.utils.default import set_default
from sparkle.src.agent.base    import base_agent

###############################################
### Particle swarm optimization
class pso(base_agent):
    def __init__(self, path, spaces, pms):
        super().__init__(path, spaces, pms)

        self.name        = "PSO"
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_points    = set_default("n_points", 20, pms)
        self.v0          = set_default("v0", 0.1, pms)
        self.c1          = set_default("c1", 0.5, pms)
        self.c2          = set_default("c2", 0.5, pms)
        self.w           = set_default("w", 0.8, pms)

        self.n_steps_total = self.n_steps_max*self.n_points

        self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Local best point
        self.p_best  = np.zeros((self.n_points, self.dim))
        self.p_score = np.ones(self.n_points)*1.0e8

    # Sample points
    # A local copy of x is required as sample() does not take previous samples as argument
    def sample(self):

        if (self.stp == 0):
            self.x = np.random.rand(self.n_points, self.dim)
            self.x = self.xmin + self.x*(self.xmax-self.xmin)
            self.v = np.random.randn(self.n_points, self.dim)*self.v0
        else:
            for i in range(self.n_points):
                r1, r2       = np.random.rand(2)
                self.v[i,:]  = (self.w*self.v[i,:] +
                                self.c1*r1*(self.p_best[i,:] - self.x[i,:]) +
                                self.c2*r2*(self.best_x[:]   - self.x[i,:]))
                self.x[i,:] += self.v[i,:]

        return self.x

    # Step
    def step(self, x, c):

        # Update best
        self.update_best(x, c)
        self.update_local_best(x, c)

        # Store
        self.store(x, c)

        self.stp += 1

    # Update local best
    def update_local_best(self, x, c):

        for i in range(self.n_points):

            # Update best local score
            if (c[i] <= self.p_score[i]):
                self.p_score[i]  = c[i]
                self.p_best[i,:] = x[i,:]

