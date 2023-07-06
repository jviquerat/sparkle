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

        self.timer_env.tic()

        # Send
        data = [('step', None)]*mpi.size
        for p in range(mpi.size):
            xp = x[p]
            data[p] = ('cost', p)
        mpi.comm.scatter(data, root=0)

        # Main process executing
        c = self.worker.cost(data[0][1])

        # Receive
        costs = np.empty((mpi.size))

        data = mpi.comm.gather((c), root=0)

        for p in range(mpi.size):
            c        = data[p]
            costs[p] = c

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
    def render(self, render):

        pass

    #     # Not all environments will render simultaneously
    #     # We use a list to store those that render and those that don't
    #     rnd = [[] for _ in range(mpi.size)]

    #     # Send
    #     data = [('render',False) for i in range(mpi.size)]
    #     for cpu in range(mpi.size):
    #         if (render[cpu]):
    #             data[cpu] = ('render',True)
    #     mpi.comm.scatter(data, root=0)

    #     # Main process executing
    #     r = self.worker.render(data[0][1])

    #     # Receive
    #     data = mpi.comm.gather(r, root=0)
    #     for cpu in range(mpi.size):
    #         if (render[cpu]):
    #             rnd[cpu] = data[cpu]

    #     return rnd

    # Close
    def close(self):

        data = [('close',None) for i in range(mpi.size)]
        data = mpi.comm.scatter(data, root=0)

        # Main process executing
        self.worker.close()
