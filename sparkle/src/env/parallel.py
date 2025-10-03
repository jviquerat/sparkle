import os
import sys
from types import SimpleNamespace
from typing import Any

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize   = False
from mpi4py import MPI

from sparkle.src.utils.default import set_default

###############################################
class SpkParallel:
    """
    A wrapper class for managing MPI parallelism.
    This class handles a two-level parallelism that can be used to pass
    a different sub-communicator to each environment, in order to have
    parallel environments, with eacg environment also running in parallel.
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

        # Default value for number of procs per env
        self.n_procs_per_env_ = set_default("n_procs_per_env", 1, pms)

        if not MPI.Is_initialized():
            MPI.Init()

        # Parallel data from COMM_WORLD
        self.world_comm_ = MPI.COMM_WORLD
        self.world_rank_ = MPI.COMM_WORLD.Get_rank()
        self.world_size_ = MPI.COMM_WORLD.Get_size()

        print(f"COMM_WORLD, size {self.world_size_}, rank {self.world_rank_}")

        # Deduce number of parallel envs
        self.n_envs_ = self.world_size_ // self.n_procs_per_env_

        # First communicator between root processes of each env
        # If n_procs_per_env_ is 1, this includes all processes
        self.main_color_ = MPI.UNDEFINED
        if self.world_rank_ % self.n_procs_per_env_ == 0:
            self.main_color_ = 0

        self.main_comm_ = self.world_comm_.Split(self.main_color_, 0)
        if self.world_rank_ % self.n_procs_per_env_ == 0:
            self.main_rank_ = self.main_comm_.Get_rank()
            self.main_size_ = self.main_comm_.Get_size()

            print(f"COMM_MAIN, size {self.main_size_}, local rank {self.main_rank_}, global rank {self.world_rank_}")

        # Second communicator specific to each env
        self.env_color_ = self.world_rank_ // self.n_procs_per_env_
        self.env_comm_  = self.world_comm_.Split(self.env_color_, 0)
        self.env_rank_  = self.env_comm_.Get_rank()
        self.env_size_  = self.env_comm_.Get_size()

        print(f"COMM_ENV, size {self.env_size_}, local rank {self.env_rank_}, global rank {self.world_rank_}")

        # MPI.Finalize()
        # exit(0)

    @property
    def size(self) -> int:
        """
        Returns the total number of parallel processes in the world communicator.
        """

        return self.world_size_

    @property
    def is_root(self) -> bool:
        """
        Checks if the current process is the root process in the world communicator.

        Returns:
            True if the current process is the root, False otherwise.
        """

        return self.world_rank_ == 0

    @property
    def rank(self) -> int:
        """
        Returns the rank of the current process in the world communicator.
        """

        return self.world_rank_

    @property
    def comm(self) -> Any:
        """
        Returns the world MPI communicator.
        """

        return self.world_comm_

    @property
    def n_envs(self) -> int:
        """
        Returns the number of parallel environments.
        """

        return self.n_envs_

    @property
    def n_procs_per_env(self) -> int:
        """
        Returns the number of procs per parallel environment.
        """

        return self.n_procs_per_env_

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
