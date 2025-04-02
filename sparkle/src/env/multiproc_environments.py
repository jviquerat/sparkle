import multiprocessing as mp

import numpy as np

from sparkle.src.env.base import BaseParallelEnvironments
from sparkle.src.env.multiproc_worker import MultiprocWorker
from sparkle.src.env.parallel import parallel
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default
from sparkle.src.utils.timer import Timer


###############################################
class MultiprocEnvironments(BaseParallelEnvironments):
    """
    A wrapper class for multiprocessing parallel environments.

    This class manages a set of environments running in parallel using the
    multiprocessing library. It handles communication with worker processes
    and provides methods for evaluating costs, resetting, and rendering
    environments.
    """
    def __init__(self, path, pms):
        """
        Initializes the MultiprocEnvironments.

        Args:
            path: The base path for storing results.
            pms: A SimpleNamespace object containing parameters for the environments.
        """

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

    def get_spaces(self):
        """
        Retrieves the environment's search space definition.

        Returns:
            A dictionary containing the search space definition.
        """

        spaces = {"dim": self.get("dim"),
                  "x0": self.get("x0"),
                  "xmin": self.get("xmin"),
                  "xmax": self.get("xmax"),
                  "vmin": self.get("vmin"),
                  "vmax": self.get("vmax"),
                  "levels": self.get("levels")}

        return spaces

    def get(self, name):
        """
        Retrieves a specific quantity from the environment.

        Args:
            name: The name of the quantity to retrieve.

        Returns:
            The value of the requested quantity.
        """

        self.pipes[0].send((name, None))
        rcv = self.pipes[0].recv()

        return rcv

    def cost(self, x):
        """
        Computes the cost of multiple points in parallel.

        Args:
            x: A NumPy array of points to evaluate.

        Returns:
            A NumPy array of the corresponding costs.
        """

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

    def reset(self, run):
        """
        Resets the environments for a new run.

        Args:
            run: The run number.

        Returns:
            A NumPy array of boolean values indicating the success of the reset operation.
        """

        # Send
        for p in self.pipes:
            p.send(('reset', run))

        # Receive
        data = np.array([])
        for p in self.pipes:
            r    = p.recv()
            data = np.append(data, r)

        return data

    def render(self, x, c):
        """
        Renders the environment.

        Args:
            x: The point to render.
            c: The cost value at the point.
        """

        # Send
        self.pipes[0].send(('render', [x,c]))

        # Receive
        rnd = self.pipes[0].recv()

        return rnd

    def close(self):
        """
        Closes the environments.
        """

        # Close all envs
        for p in self.pipes:
            p.send(('close', None))
        for p in self.procs:
            p.terminate()
            p.join()
