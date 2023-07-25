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
                #error("spk_parallel", "__init__",
                #      "Multiprocessing requires argument n_env")
                print("Error: multiprocessing requires argument n_env")
                exit(1)
            else:
                self._size = pms.n_env

        if (self._type == "mpi"):
            import mpi4py
            mpi4py.rc.initialize = False
            mpi4py.rc.finalize   = False
            from mpi4py import MPI

            if (not MPI.Is_initialized()):
                MPI.Init()
                self._comm = MPI.COMM_WORLD
                self._rank = MPI.COMM_WORLD.Get_rank()
                self._size = MPI.COMM_WORLD.Get_size()

    def type(self):

        return self._type

    def is_root(self):

        if (self._type == "mpi"):
            return (self._rank == 0)

        if (self._type == "multiprocessing"):
            return True

    def comm(self):

        if (self._type == "mpi"):
            return self._comm

        if (self._type == "multiprocessing"):
            #error("spk_parallel", "comm",
            #      "comm() is not defined for multiprocessing")
            print("Error: comm() is not defined for multiprocessing")
            exit(1)

    def size(self):

        return self._size

    def rank(self):

        if (self._type == "mpi"):
            return self._rank

        if (self._type == "multiprocessing"):
            #error("spk_parallel", "rank",
            #      "rank() is not defined for multiprocessing")
            print("Error: rank() is not defined for multiprocessing")
            exit(1)

    def environments(self, path, pms):

        if (self._type == "mpi"):
            from sparkle.src.env.mpi_environments import mpi_environments
            return mpi_environments(path, pms)

        if (self._type == "multiprocessing"):
            from sparkle.src.env.multiproc_environments import multiproc_environments
            return multiproc_environments(path, pms)

    def finalize(self):

        if (self._type == "mpi"):
            import mpi4py
            from mpi4py import MPI

            MPI.Finalize()
            exit(0)

        if (self._type == "multiprocessing"):
            pass

# Single instance
parallel = spk_parallel()
