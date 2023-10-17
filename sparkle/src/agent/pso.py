# Custom imports
from sparkle.src.agent.base import *

###############################################
### Particle swarm optimization
class pso(base_agent):
    def __init__(self, path, dim, xmin, xmax, pms):

        super().__init__(pms)

        self.name        = "PSO"
        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.n_steps_max = 20
        self.n_points    = 20
        self.v0          = 0.1
        self.c1          = 0.5
        self.c2          = 0.5
        self.w           = 0.8

        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        if hasattr(pms, "n_points"):    self.n_points    = pms.n_points
        if hasattr(pms, "v0"):          self.v0          = pms.v0
        if hasattr(pms, "c1"):          self.c1          = pms.c1
        if hasattr(pms, "c2"):          self.c2          = pms.c2
        if hasattr(pms, "w"):           self.w           = pms.w

        self.n_steps_total = self.n_steps_max*self.n_points

        self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Positions and velocities
        self.x = np.random.rand(self.n_points, self.dim)
        self.x = self.xmin + self.x*(self.xmax-self.xmin)
        self.v = np.random.randn(self.n_points, self.dim)*self.v0

        # Local best point
        self.p_best  = np.copy(self.x)
        self.p_score = np.ones(self.n_points)*1.0e8

        return self.x

    # Step
    # Data storage is performed between update of best points
    # and update of positions and velocities so the recorded (x,v)
    # matches with the correct cost
    def step(self, c):

        # Update best
        self.update_best(c)
        self.update_local_best(c)

        # Store
        self.store(c)

        # Update position and velocities
        self.update_xv()

        self.stp += 1

    # Update local best
    def update_local_best(self, c):

        for i in range(self.n_points):

            # Update best local score
            if (c[i] <= self.p_score[i]):
                self.p_score[i]  = c[i]
                self.p_best[i,:] = self.x[i,:]

    # Update positions and velocities
    def update_xv(self):

        for i in range(self.n_points):
            r1, r2 = np.random.rand(2)
            self.v[i,:]  = (self.w*self.v[i,:] +
                            self.c1*r1*(self.p_best[i,:] - self.x[i,:]) +
                            self.c2*r2*(self.best_x[:]   - self.x[i,:]))
            v = np.random.randn(self.n_points, self.dim)*self.v0
            self.x[i,:] += self.v[i,:]
