# Generic imports
import os
import sys
import numpy as np
import time

# Custom imports
from sparkle.src.env.worker  import *
from sparkle.src.utils.timer import *

###############################################
### A wrapper class for parallel environments
class environments:
    def __init__(self, path, pms):

        # Default parameters
        self.name = pms.name
        self.args = None

        if hasattr(pms, "args"): self.args = pms.args

        # Generate workers
        self.worker = worker(self.name, self.args, mpi.rank, path)

        # Set all slaves to wait for instructions
        if (mpi.rank != 0):
            self.worker.work()

        # Initialize timer
        self.timer_env = timer("env      ")

    # Return dimension of environment
    def dim(self):

        if (mpi.rank == 0):
            return self.worker.env.dim

    # Compute cost in all environments
    def cost(self, x):

        # Initialize stuff
        n_dof   = x.shape[0]
        costs   = np.zeros((n_dof))
        n_loops = n_dof//mpi.size

        self.timer_env.tic()

        for i in range(n_loops):

            # Send
            data = [('step', None)]*mpi.size
            for p in range(mpi.size):
                data[p] = ('cost', x[i*mpi.size+p])
            mpi.comm.scatter(data, root=0)

            # Main process executing
            c = self.worker.cost(data[0][1])

            # Receive
            data = mpi.comm.gather((c), root=0)

            for p in range(mpi.size):
                c        = data[p]
                costs[i*mpi.size+p] = c

        self.timer_env.toc()

        return costs

    # Reset environments
    def reset(self):

        # Send
        data = [('reset', True) for i in range(mpi.size)]
        mpi.comm.scatter(data, root=0)

        # Main process executing
        r = self.worker.reset(data[0][1])

        # Receive and normalize
        results = np.empty((mpi.size))
        data    = mpi.comm.gather((r), root=0)
        for p in range(mpi.size):
            results[p] = self.process_obs(data[p])

        return results

    # Render environment
    def render(self, x):

        if (mpi.rank == 0):
            return self.worker.env.render(x)

    # Close
    def close(self):

        data = [('close',None) for i in range(mpi.size)]
        data = mpi.comm.scatter(data, root=0)

        # Main process executing
        self.worker.close()
