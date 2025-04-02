# Generic imports
import numpy as np
from numpy import ndarray
from types import SimpleNamespace
from typing import Dict, List, Union, Any

# Custom imports
from sparkle.src.env.parallel   import parallel
from sparkle.src.env.base       import BaseParallelEnvironments
from sparkle.src.env.mpi_worker import MpiWorker
from sparkle.src.env.spaces     import EnvSpaces
from sparkle.src.utils.default  import set_default
from sparkle.src.utils.timer    import Timer

###############################################
### A wrapper class for mpi parallel environments
class MpiEnvironments(BaseParallelEnvironments):
    def __init__(self, path: str, pms: SimpleNamespace) -> None:

        # Default parameters
        self.name = pms.name
        self.args = set_default("args", None, pms)

        # Generate workers
        self.worker = MpiWorker(self.name, self.args, parallel.rank(), path)

        # Set all slaves to wait for instructions
        if not parallel.is_root(): self.worker.work()

        # Declare spaces object
        self.spaces = EnvSpaces(self.get_spaces(), pms)

        # Initialize timer
        self.timer_env = Timer("env      ")

    # Get environment spaces
    def get_spaces(self) -> Any:

        spaces = {"dim": self.worker.env.dim,
                  "x0": self.worker.env.x0,
                  "xmin": self.worker.env.xmin,
                  "xmax": self.worker.env.xmax}

        if hasattr(self.worker.env, "vmin"):   spaces["vmin"]   = self.worker.env.vmin
        if hasattr(self.worker.env, "vmax"):   spaces["vmax"]   = self.worker.env.vmax
        if hasattr(self.worker.env, "levels"): spaces["levels"] = self.worker.env.levels

        return spaces

    # Compute cost in all environments
    def cost(self, x: ndarray) -> ndarray:

        # Initialize stuff
        n_dof   = x.shape[0]
        costs   = np.zeros((n_dof))
        n_loops = n_dof//parallel.size

        self.timer_env.tic()

        for i in range(n_loops):

            # Send
            data = [('step', None)]*parallel.size
            for p in range(parallel.size):
                data[p] = ('cost', x[i*parallel.size+p])
            parallel.comm().scatter(data, root=0)

            # Main process executing
            c = self.worker.cost(data[0][1])

            # Receive
            data = parallel.comm().gather((c), root=0)

            for p in range(parallel.size):
                c        = data[p]
                costs[i*parallel.size+p] = c

        self.timer_env.toc()

        return costs

    # Reset environments
    def reset(self, run: int) -> List[bool]:

        # Send
        data = [('reset', run) for i in range(parallel.size)]
        parallel.comm().scatter(data, root=0)

        # Main process executing
        r = self.worker.reset(data[0][1])

        # Receive and normalize
        data = parallel.comm().gather((r), root=0)

        return data

    # Render environment
    def render(self, x, c, **kwargs):

        if parallel.is_root():
            return self.worker.render(x, c, **kwargs)

    # Close
    def close(self) -> None:

        data = [('close',None) for i in range(parallel.size)]
        data = parallel.comm().scatter(data, root=0)

        # Main process executing
        self.worker.close()

