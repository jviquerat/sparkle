###############################################
### A wrapper class for parallelism
class SpkParallel:

    def __init__(self):

        # Default values
        self.size_ = 1
        self.type_ = None

    def set(self, pms):

        if (hasattr(pms, "parallel_type")):
            self.type_ = pms.parallel_type
        else:
            self.type_ = "mpi"

        if (self.type_ == "multiprocessing"):
            if (not hasattr(pms, "n_env")):
                print("Parallel: multiprocessing requires argument n_env")
                exit(0)
            else:
                self.size_ = pms.n_env

        if (self.type_ == "mpi"):
            import mpi4py
            mpi4py.rc.initialize = False
            mpi4py.rc.finalize   = False
            from mpi4py import MPI

            if (not MPI.Is_initialized()):
                MPI.Init()
                self._comm = MPI.COMM_WORLD
                self._rank = MPI.COMM_WORLD.Get_rank()
                self.size_ = MPI.COMM_WORLD.Get_size()

    @property
    def type(self):
        return self.type_

    @property
    def size(self):
        return self.size_

    def is_root(self):

        if (self.type_ == "mpi"):
            return (self._rank == 0)

        if (self.type_ == "multiprocessing"):
            return True

        return True

    def comm(self):

        if (self.type_ == "mpi"):
            return self._comm

        if (self.type_ == "multiprocessing"):
            print("# Parallel: comm() is not defined for multiprocessing")
            exit(0)

    def rank(self):

        if (self.type_ == "mpi"):
            return self._rank

        if (self.type_ == "multiprocessing"):
            print("# Parallel: rank() is not defined for multiprocessing")
            exit(0)

    def environments(self, path, pms):

        if (self.type_ == "mpi"):
            from sparkle.src.env.mpi_environments import MpiEnvironments
            return MpiEnvironments(path, pms)

        if (self.type_ == "multiprocessing"):
            from sparkle.src.env.multiproc_environments import MultiprocEnvironments
            return MultiprocEnvironments(path, pms)

    def finalize(self):

        if (self.type_ == "mpi"):
            from mpi4py import MPI

            MPI.Finalize()

        if (self.type_ == "multiprocessing"):
            pass

# Single instance
parallel = SpkParallel()
