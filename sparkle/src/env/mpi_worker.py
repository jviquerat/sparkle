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
            data    = None
            data    = parallel.comm().scatter(data, root=0)
            command = data[0]
            data    = data[1]

            # Execute commands
            if command == 'cost':
                c = self.cost(data)
                parallel.comm().gather((c), root=0)

            if command == 'reset':
                r = self.reset(data)
                parallel.comm().gather((r), root=0)

            if command == 'render':
                rnd = self.render(data)
                parallel.comm().gather((rnd), root=0)

            if command == 'close':
                self.close()
                parallel.finalize()
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
