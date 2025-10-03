import sys

from numpy import float64, ndarray

from sparkle.src.env.parallel import parallel


###############################################
class MpiWorker():
    """
    Worker class for MPI slave processes.

    This class defines the behavior of a worker process in an MPI-based
    parallel environment. It handles communication with the main process and
    executes commands related to environment interaction.
    """
    def __init__(self, env_name: str, args: None, cpu: int, path: str) -> None:
        """
        Initializes the MpiWorker.

        Args:
            env_name: The name of the environment module.
            args: Additional arguments for the environment constructor.
            cpu: The CPU index for the worker.
            path: The base path for storing results.
        """

        # Build environment
        module    = __import__(env_name)
        env_build = getattr(module, env_name)

        if parallel.n_procs_per_env > 1:
            if args is not None:
                self.env = env_build(cpu, path, parallel.env_comm, args)
            else:
                self.env = env_build(cpu, path, parallel.env_comm)
        else:
            if args is not None:
                self.env = env_build(cpu, path, args)
            else:
                self.env = env_build(cpu, path)

    def work(self):
        """
        The main working loop for slave processes.

        This method continuously receives commands from the main process,
        executes them, and sends back the results.
        """
        while True:
            data = None

            # env_root procs receive from world root on main_comm
            if parallel.is_env_root:
                data = parallel.main_comm.scatter(data, root=0)

            # env_root procs scatter to env procs on their own comm
            data = parallel.env_comm.bcast(data, root=0)

            # Extract command data
            command = data[0]
            values  = data[1]

            # Execute commands
            if command == 'cost':
                c = self.cost(values)
                if parallel.is_env_root:
                    parallel.main_comm.gather((c), root=0)

            if command == 'reset':
                r = self.reset(values)
                if parallel.is_env_root:
                    parallel.main_comm.gather((r), root=0)

            if command == 'render':
                rnd = self.render(values)
                if parallel.is_env_root:
                    parallel.main_comm.gather((rnd), root=0)

            if command == 'close':
                self.close()
                sys.exit(0)

    def cost(self, x: ndarray) -> float64:
        """
        Computes the cost of a point in the environment.

        Args:
            x: The point to evaluate.

        Returns:
            The cost value.
        """

        return self.env.cost(x)

    def validate(self, x: ndarray) -> float64:
        """
        Validate a point against a priori environment constraints
        XXX This function is directly called using the main process

        Args:
            x: The point to evaluate

        Returns:
            True if point is valid, False otherwise
        """

        if hasattr(self.env, "validate"):
            return self.env.validate(x)
        else:
            return True

    def reset(self, run: int) -> bool:
        """
        Resets the environment for a new run.

        Args:
            run: The run number.

        Returns:
            A boolean value indicating the success of the reset operation.
        """

        return self.env.reset(run)

    def render(self, x, c, **kwargs):
        """
        Renders the environment.

        Args:
            x: The point to render.
            c: The cost value at the point.
            **kwargs: Additional keyword arguments for rendering.
        """

        return self.env.render(x, c, **kwargs)

    def close(self) -> None:
        """
        Closes the environment.
        """
        self.env.close()
        parallel.finalize()
