import numpy           as np
import multiprocessing as mp

from sparkle.src.env.parallel         import parallel
from sparkle.src.env.base             import BaseParallelEnvironments
from sparkle.src.env.multiproc_worker import MultiprocWorker
from sparkle.src.env.spaces           import EnvSpaces
from sparkle.src.utils.default        import set_default
from sparkle.src.utils.timer          import Timer

###############################################
### A wrapper class for multiprocessing parallel environments
class MultiprocEnvironments(BaseParallelEnvironments):
    def __init__(self, path, pms):

        # Default parameters
        self.name  = pms.name
        self.args  = set_default("args", None, pms)
        self.pipes = []
        self.procs = []

        # Start environments
        for env in range(parallel.size):
            p_pipe, c_pipe = mp.Pipe()
            process        = mp.Process(target = MultiprocWorker,
                                        args   = (self.name, self.args,
                                                  env, path, c_pipe))

            self.pipes.append(p_pipe)
            self.procs.append(process)

            process.daemon = True
            process.start()

        # Declare spaces object
        self.spaces = EnvSpaces(self.get_spaces(), pms)

        # Initialize timer
        self.timer_env = Timer("env      ")

    # Get environment spaces
    def get_spaces(self):

        spaces = {"dim": self.get("dim"),
                  "x0": self.get("x0"),
                  "xmin": self.get("xmin"),
                  "xmax": self.get("xmax"),
                  "vmin": self.get("vmin"),
                  "vmax": self.get("vmax"),
                  "levels": self.get("levels")}

        return spaces

    # Get quantity based on name
    def get(self, name):

        self.pipes[0].send((name, None))
        rcv = self.pipes[0].recv()

        return rcv

    # Compute cost in all environments
    def cost(self, x):

        # Initialize stuff
        n_dof   = x.shape[0]
        costs   = np.zeros((n_dof))
        n_loops = n_dof//parallel.size

        self.timer_env.tic()

        for i in range(n_loops):

            # Send
            for p in range(parallel.size):
                self.pipes[p].send(('cost', x[i*parallel.size+p]))

            # Receive
            for p in range(parallel.size):
                c = self.pipes[p].recv()
                costs[i*parallel.size+p] = c

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
