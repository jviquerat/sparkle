import sys
from types import SimpleNamespace
from typing import Any


###############################################
class SpkParallel:
    """
    A wrapper class for managing parallelism.

    This class provides a unified interface for handling different types of
    parallelism, such as MPI and multiprocessing. It allows for easy
    configuration and management of parallel environments.
    """

    def __init__(self) -> None:
        """
        Initializes the SpkParallel.
        """

        # Default values
        self.size_ = 1
        self.type_ = None

    def set(self, pms: SimpleNamespace) -> None:
        """
        Configures the parallelism settings.

        Args:
            pms: A SimpleNamespace object containing parameters for parallelism.
        """

        if hasattr(pms, "parallel_type"):
            self.type_ = pms.parallel_type
        else:
            self.type_ = "mpi"

        if self.type_ == "multiprocessing":
            if not hasattr(pms, "n_env"):
                print("Parallel: multiprocessing requires argument n_env")
                sys.exit(1)
            else:
                self.size_ = pms.n_env

        if self.type_ == "mpi":
            import mpi4py
            mpi4py.rc.initialize = False
            mpi4py.rc.finalize   = False
            from mpi4py import MPI

            if not MPI.Is_initialized():
                MPI.Init()
                self._comm = MPI.COMM_WORLD
                self._rank = MPI.COMM_WORLD.Get_rank()
                self.size_ = MPI.COMM_WORLD.Get_size()

    @property
    def type(self):
        """
        Returns the type of parallelism.
        """
        return self.type_

    @property
    def size(self) -> int:
        """
        Returns the number of parallel processes.
        """
        return self.size_

    def is_root(self) -> bool:
        """
        Checks if the current process is the root process.

        Returns:
            True if the current process is the root, False otherwise.
        """

        if self.type_ == "mpi":
            return self._rank == 0

        if self.type_ == "multiprocessing":
            return True

        return True

    def comm(self) -> Any:
        """
        Returns the MPI communicator.

        Returns:
            The MPI communicator.

        Raises:
            SystemExit: If called in multiprocessing mode.
        """

        if self.type_ == "mpi":
            return self._comm

        if self.type_ == "multiprocessing":
            print("# Parallel: comm() is not defined for multiprocessing")
            sys.exit(1)

    def rank(self) -> int:
        """
        Returns the rank of the current process.

        Returns:
            The rank of the current process.

        Raises:
            SystemExit: If called in multiprocessing mode.
        """

        if self.type_ == "mpi":
            return self._rank

        if self.type_ == "multiprocessing":
            print("# Parallel: rank() is not defined for multiprocessing")
            sys.exit(1)

    def environments(self, path: str, pms: SimpleNamespace) -> Any:
        """
        Creates and returns the parallel environments.

        Args:
            path: The base path for storing results.
            pms: A SimpleNamespace object containing parameters for the environments.

        Returns:
            An instance of the parallel environments.
        """

        if self.type_ == "mpi":
            from sparkle.src.env.mpi_environments import MpiEnvironments
            return MpiEnvironments(path, pms)

        if self.type_ == "multiprocessing":
            from sparkle.src.env.multiproc_environments import MultiprocEnvironments
            return MultiprocEnvironments(path, pms)

    def finalize(self) -> None:
        """
        Finalizes the parallelism.
        """

        if self.type_ == "mpi":
            from mpi4py import MPI

            MPI.Finalize()

        if self.type_ == "multiprocessing":
            pass

# Single instance
parallel = SpkParallel()
