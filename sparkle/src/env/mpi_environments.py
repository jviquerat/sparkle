# Generic imports
import os
import sys
import numpy as np
import time

# Custom imports
from sparkle.src.env.mpi_worker  import *
from sparkle.src.utils.timer import *

###############################################
### A wrapper class for mpi parallel environments
class mpi_environments:
    def __init__(self, path, pms):

        # Default parameters
        self.name = pms.name
        self.args = None

        if hasattr(pms, "args"): self.args = pms.args

        # Generate workers
        self.worker = mpi_worker(self.name, self.args, parallel.rank(), path)

        # Set all slaves to wait for instructions
        if (not parallel.is_root()):
            self.worker.work()

        # Initialize timer
        self.timer_env = timer("env      ")

    # Return dimension of environment
    def dim(self):

        if (parallel.is_root()):
            return self.worker.env.dim

    # Return xmin value
    def xmin(self):

        if (parallel.is_root()):
            return self.worker.env.xmin

    # Return xmax value
    def xmax(self):

        if (parallel.is_root()):
            return self.worker.env.xmax

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
    def render(self, x):

        if (parallel.is_root()):
            return self.worker.env.render(x)

    # Close
    def close(self):

        data = [('close',None) for i in range(parallel.size())]
        data = parallel.comm().scatter(data, root=0)

        # Main process executing
        self.worker.close()
