# Custom imports
from sparkle.src.agent.base import *

###############################################
### CEM
class cem(base_agent):
    def __init__(self, path, dim, xmin, xmax, pms):

        self.name        = "CEM"
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

        self.n_steps_total = self.n_steps_max*self.n_points

        self.summary()

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

        # Data storage
        self.hist_t = np.zeros((self.n_steps_total))           # time
        self.hist_c = np.zeros((self.n_steps_total))           # cost
        self.hist_b = np.zeros((self.n_steps_total))           # best cost
        self.hist_x = np.zeros((self.n_steps_total, self.dim)) # dofs

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
