# Generic imports
import numpy as np

# Custom imports
from sparkle.src.env.parallel import parallel
from sparkle.src.utils.error  import error

###############################################
### A base class for parallel environments
class BaseParallelEnvironments():
    def __init__(self):
        pass

    # Evaluate cost on a list of points in parallel
    def evaluate(self, x):

        n_points = x.shape[0]

        if (n_points%parallel.size != 0):
            error("base_parallel_environments", "evaluate",
                  "nb of evaluation pts should be a multiple of nb of parallel envs")

        n_steps = n_points//parallel.size
        costs   = np.zeros(n_points)

        step = 0
        while (step < n_steps):
            end = "\r"
            if (step == n_steps-1): end = "\n"
            i_start = step*parallel.size
            i_end   = (step+1)*parallel.size - 1
            print("# Computing individuals #"+str(i_start)+" to #"+str(i_end), end=end)

            xp = np.zeros((parallel.size, self.spaces.dim))
            for k in range(parallel.size):
                xp[k,:] = x[step*parallel.size + k]

            c = self.cost(xp)
            for k in range(parallel.size):
                costs[step*parallel.size + k] = c[k]

            step += 1

        return costs

    # Generate cost map for rendering 1D envs
    def generate_cost_map_1D(self):

        n_plot   = 400
        x_plot   = np.linspace(self.spaces.xmin[0], self.spaces.xmax[0], num=n_plot)
        cost_map = np.zeros(n_plot)

        for i in range(n_plot):
            x = np.array([[x_plot[i]]])
            cost_map[i] = self.cost(x)

        return x_plot, cost_map

    # Generate cost map for rendering 2D envs
    def generate_cost_map_2D(self):

        n_plot   = 100
        x_plot   = np.linspace(self.spaces.xmin[0], self.spaces.xmax[0], num=n_plot)
        y_plot   = np.linspace(self.spaces.xmax[1], self.spaces.xmin[1], num=n_plot)
        grid     = np.array(np.meshgrid(x_plot, y_plot))
        x_plot   = grid[0]
        y_plot   = grid[1]
        cost_map = np.zeros((n_plot,n_plot))

        for i in range(n_plot):
            for j in range(n_plot):
                x = np.array([[x_plot[i,j], y_plot[i,j]]])
                cost_map[i,j] = self.cost(x)[0]

        return x_plot, y_plot, cost_map
