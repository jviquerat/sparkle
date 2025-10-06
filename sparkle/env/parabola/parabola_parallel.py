import numpy as np

import mpi4py
from mpi4py import MPI

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default


###############################################
class parabola_parallel(base_env):
    """
    Standard parabola function in arbitrary dimension
    """
    def __init__(self, cpu, path, comm, pms=None):

        # Fill structure
        self.name      = 'parabola_parallel'
        self.base_path = path
        self.cpu       = cpu
        self.comm      = comm
        self.par_size  = comm.Get_size()
        self.par_rank  = comm.Get_rank()

        self.dim       = set_default("dim",  self.par_size, pms)
        self.x0        = set_default("x0",   2.5*np.ones(self.dim), pms)
        self.xmin      = set_default("xmin",-5.0*np.ones(self.dim), pms)
        self.xmax      = set_default("xmax", 5.0*np.ones(self.dim), pms)

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 20.0
        self.levels    = [0.1, 1.0, 5.0, 10.0, 20.0]

    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    def cost(self, x):

        v    = x[self.par_rank]**2
        vsum = self.comm.reduce(v, op=MPI.SUM, root=0)

        if self.par_rank == 0:
            return vsum
        else:
            return v

    def close(self):
        pass
