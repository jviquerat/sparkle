# Generic imports
import os
import sys
import numpy as np

# Custom imports
from sparkle.src.env.parallel   import parallel
from sparkle.src.env.mpi_worker import mpi_worker
from sparkle.src.env.spaces     import environment_spaces
from sparkle.src.utils.timer    import timer

###############################################
### A wrapper class for mpi parallel environments
class mpi_environments:
    def __init__(self, path, pms):

        # Default parameters
        self.name = pms.name
        self.args = None

        # Optional parameters
        if hasattr(pms, "args"): self.args = pms.args

        # Generate workers
        self.worker = mpi_worker(self.name, self.args, parallel.rank(), path)

        # Set all slaves to wait for instructions
        if (not parallel.is_root()): self.worker.work()

        # Declare spaces object
        self.spaces = environment_spaces(self.get_spaces(), pms)

        # Initialize timer
        self.timer_env = timer("env      ")

    # Get environment spaces
    def get_spaces(self):

        spaces = [self.worker.env.dim,
                  self.worker.env.x0,
                  self.worker.env.xmin,
                  self.worker.env.xmax]

        if hasattr(self.worker.env, "vmin"):   spaces += [self.worker.env.vmin]
        if hasattr(self.worker.env, "vmax"):   spaces += [self.worker.env.vmax]
        if hasattr(self.worker.env, "levels"): spaces += [self.worker.env.levels]

        return spaces

    # Compute cost in all environments
    def cost(self, x):

        # Initialize stuff
        n_dof   = x.shape[0]
        costs   = np.zeros((n_dof))
        n_loops = n_dof//parallel.size()

        self.timer_env.tic()

        for i in range(n_loops):

            # Send
            data = [('step', None)]*parallel.size()
            for p in range(parallel.size()):
                data[p] = ('cost', x[i*parallel.size()+p])
            parallel.comm().scatter(data, root=0)

            # Main process executing
            c = self.worker.cost(data[0][1])

            # Receive
            data = parallel.comm().gather((c), root=0)

            for p in range(parallel.size()):
                c        = data[p]
                costs[i*parallel.size()+p] = c

        self.timer_env.toc()

        return costs

    # Reset environments
    def reset(self, run):

        # Send
        data = [('reset', run) for i in range(parallel.size())]
        parallel.comm().scatter(data, root=0)

        # Main process executing
        r = self.worker.reset(data[0][1])

        # Receive and normalize
        data = parallel.comm().gather((r), root=0)

        return data

    # Render environment
    def render(self, x, c, **kwargs):

        if (parallel.is_root()):
            return self.worker.render(x, c, **kwargs)

    # Close
    def close(self):

        data = [('close',None) for i in range(parallel.size())]
        data = parallel.comm().scatter(data, root=0)

        # Main process executing
        self.worker.close()
