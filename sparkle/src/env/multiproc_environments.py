# Generic imports
import os
import sys
import numpy           as np
import multiprocessing as mp

# Custom imports
from sparkle.src.env.parallel         import parallel
from sparkle.src.env.multiproc_worker import multiproc_worker
from sparkle.src.env.spaces           import environment_spaces
from sparkle.src.utils.timer          import timer

###############################################
### A wrapper class for multiprocessing parallel environments
class multiproc_environments:
    def __init__(self, path, pms):

        # Default parameters
        self.name  = pms.name
        self.args  = None
        self.pipes = []
        self.procs = []

        # Optional arguments to pass to environments
        if hasattr(pms, "args"): self.args = pms.args

        # Start environments
        for env in range(parallel.size()):
            p_pipe, c_pipe = mp.Pipe()
            process        = mp.Process(target = multiproc_worker,
                                        args   = (self.name, self.args,
                                                  env, path, c_pipe))

            self.pipes.append(p_pipe)
            self.procs.append(process)

            process.daemon = True
            process.start()

        # Declare spaces object
        self.spaces = environment_spaces(self.get_spaces(), pms)

        # Initialize timer
        self.timer_env = timer("env      ")

    # Get environment spaces
    def get_spaces(self):

        return [self.dim(), self.x0(), self.xmin(), self.xmax()]

    # Return dimension of environment
    def dim(self):

        # Send
        self.pipes[0].send(('dim', None))

        # Receive
        d = self.pipes[0].recv()

        return d

    # Return x0 value
    def x0(self):

        # Send
        self.pipes[0].send(('x0', None))

        # Receive
        x0 = self.pipes[0].recv()

        return x0

    # Return xmin value
    def xmin(self):

        # Send
        self.pipes[0].send(('xmin', None))

        # Receive
        xm = self.pipes[0].recv()

        return xm

    # Return xmax value
    def xmax(self):

        # Send
        self.pipes[0].send(('xmax', None))

        # Receive
        xm = self.pipes[0].recv()

        return xm

    # Compute cost in all environments
    def cost(self, x):

        # Initialize stuff
        n_dof   = x.shape[0]
        costs   = np.zeros((n_dof))
        n_loops = n_dof//parallel.size()

        self.timer_env.tic()

        for i in range(n_loops):

            # Send
            for p in range(parallel.size()):
                self.pipes[p].send(('cost', x[i*parallel.size()+p]))

            # Receive
            for p in range(parallel.size()):
                c = self.pipes[p].recv()
                costs[i*parallel.size()+p] = c

        self.timer_env.toc()

        return costs

    # Reset environments
    def reset(self, run):

        # Send
        for p in self.pipes:
            p.send(('reset', run))

        # Receive
        data = np.array([])
        for p in self.pipes:
            r    = p.recv()
            data = np.append(data, r)

        return data

    # Render environment
    def render(self, x, c):

        # Send
        self.pipes[0].send(('render', [x,c]))

        # Receive
        rnd = self.pipes[0].recv()

        return rnd

    # Close
    def close(self):

        # Close all envs
        for p in self.pipes:
            p.send(('close', None))
        for p in self.procs:
            p.terminate()
            p.join()
