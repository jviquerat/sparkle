# Generic imports
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize   = False
from mpi4py import MPI

# Custom imports
from sparkle.src.utils.error  import *

###############################################
### A wrapper class for parallelism
class spk_parallel:

    def __init__(self):
        pass

    def set(self, pms):

        if (hasattr(pms, "parallel_type")):
            self._type = pms.parallel_type
        else:
            self._type = "mpi"

        if (self._type == "multiprocessing"):
            if (not hasattr(pms, "n_env")):
                error("spk_parallel", "__init__",
                      "Multiprocessing requires argument n_env")
            else:
                self._size = pms.n_env

        if (self._type == "mpi"):
            if (not MPI.Is_initialized()):
                MPI.Init()
                self._comm = MPI.COMM_WORLD
                self._rank = MPI.COMM_WORLD.Get_rank()
                self._size = MPI.COMM_WORLD.Get_size()

    def is_root(self):

        if (self._type == "mpi"):
            return (self._rank == 0)

        if (self._type == "multiprocessing"):
            return True

    def comm(self):

        if (self._type == "mpi"):
            return self._comm

        if (self._type == "multiprocessing"):
            error("spk_parallel", "comm",
                  "comm() is not defined for multiprocessing")

    def size(self):

        return self._size

    def rank(self):

        if (self._type == "mpi"):
            return self._rank

        if (self._type == "multiprocessing"):
            error("spk_parallel", "rank",
                  "rank() is not defined for multiprocessing")

    def environments(self, path, pms):

        if (self._type == "mpi"):
            from sparkle.src.env.mpi_environments import mpi_environments
            return mpi_environments(path, pms)

        if (self._type == "multiprocessing"):
            pass

    def finalize(self):

        if (self._type == "mpi"):
            MPI.Finalize()

        if (self._type == "multiprocessing"):
            pass

# Single instance
parallel = spk_parallel()
