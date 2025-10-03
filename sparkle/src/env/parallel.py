import os
import sys
from types import SimpleNamespace
from typing import Any

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize   = False
from mpi4py import MPI

###############################################
class SpkParallel:
    """
    A wrapper class for managing MPI parallelism.
    """

    def __init__(self) -> None:
        """
        Initializes the SpkParallel.
        """

        self.size_ = 1
        self.n_envs_ = 1

    def set(self, pms: SimpleNamespace) -> None:
        """
        Configures the parallelism settings.

        Args:
            pms: A SimpleNamespace object containing parameters for parallelism.
        """

        if not MPI.Is_initialized():
            MPI.Init()
            self.comm_   = MPI.COMM_WORLD
            self.rank_   = MPI.COMM_WORLD.Get_rank()
            self.size_   = MPI.COMM_WORLD.Get_size()
            self.n_envs_ = self.size_

    @property
    def size(self) -> int:
        """
        Returns the total number of parallel processes.
        """

        return self.size_

    @property
    def n_envs(self) -> int:
        """
        Returns the number of parallel environments
        """

        return self.n_envs_

    @property
    def is_root(self) -> bool:
        """
        Checks if the current process is the root process.

        Returns:
            True if the current process is the root, False otherwise.
        """

        return self.rank_ == 0

    def comm(self) -> Any:
        """
        Returns the MPI communicator.
        """

        return self.comm_

    def rank(self) -> int:
        """
        Returns the rank of the current process.
        """

        return self.rank_

    def environments(self, path: str, pms: SimpleNamespace) -> Any:
        """
        Creates and returns the parallel environments.

        Args:
            path: The base path for storing results.
            pms: A SimpleNamespace object containing parameters for the environments.

        Returns:
            An instance of the parallel environments.
        """

        from sparkle.src.env.mpi_environments import MpiEnvironments
        return MpiEnvironments(path, pms)

    def finalize(self) -> None:
        """
        Finalizes the parallelism.
        """
        # Do not finalize if running under pytest, as it runs all tests
        # in a single process. MPI will be finalized when the process exits.
        if "PYTEST_CURRENT_TEST" in os.environ:
            return

        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()

# Single instance
parallel = SpkParallel()
