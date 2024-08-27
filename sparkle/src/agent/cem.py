# Custom imports
from sparkle.src.agent.base import *

###############################################
### CEM
class cem(base_agent):
    def __init__(self, path, dim, x0, xmin, xmax, pms):

        super().__init__(pms)

        self.name        = "CEM"
        self.base_path   = path
        self.dim         = dim
        self.x0          = x0
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

        self.n_steps_total = self.n_steps_max*self.n_points

        self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Min and max arrays used for cem adaptation
        self.xmin_cem = self.xmin.copy()
        self.xmax_cem = self.xmax.copy()

    # Sample from distribution
    def sample(self):

        x = np.zeros((self.n_points, self.dim))

        for i in range(self.n_points):
            x[i,:] = np.random.uniform(low  = self.xmin_cem,
                                       high = self.xmax_cem)

        return x

    # Step
    def step(self, x, c):

        # Update best value
        self.update_best(x, c)

        # Sort
        self.sort(x, c)

        # Store
        self.store(x, c)

        # Update xmin and xmax
        xmin = np.amin(x[:self.n_elites,:], axis=0)
        xmax = np.amax(x[:self.n_elites,:], axis=0)
        self.xmin_cem[:] = ((1.0-self.alpha)*self.xmin_cem[:] + self.alpha*xmin[:])
        self.xmax_cem[:] = ((1.0-self.alpha)*self.xmax_cem[:] + self.alpha*xmax[:])

        self.stp += 1

    # Sort offsprings based on cost
    # x and c arrays are actually modified here
    def sort(self, x, c):

        sc   = np.argsort(c)
        x[:] = x[sc[:]]
        c[:] = c[sc[:]]
