from types import SimpleNamespace

from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.prints import spacer


###############################################

class BaseAgent():
    """
    Base class for all agents.

    This class defines the common interface and functionality for all agents
    used in the optimization framework. It provides methods for resetting,
    sampling, stepping, rendering, and summarizing agent information.
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: SimpleNamespace) -> None:
        """
        Initializes the BaseAgent.

        Args:
            path: The base path for storing results.
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the agent.
        """

        self.spaces    = spaces
        self.base_path = path

        self.silent = False
        if hasattr(pms, "silent"): self.silent = pms.silent

    @property
    def dim(self) -> int:
        """
        Returns the dimensionality of the search space.
        """
        return self.spaces.dim

    @property
    def natural_dim(self) -> int:
        """
        Returns the natural dimensionality of the search space.
        """
        return self.spaces.natural_dim

    @property
    def x0(self) -> ndarray:
        """
        Returns the initial point in the search space.
        """
        return self.spaces.x0

    @property
    def xmin(self) -> ndarray:
        """
        Returns the lower bounds of the search space.
        """
        return self.spaces.xmin

    @property
    def xmax(self) -> ndarray:
        """
        Returns the upper bounds of the search space.
        """
        return self.spaces.xmax

    def reset(self, run: int) -> None:
        """
        Resets the agent for a new run.

        Args:
            run: The run number.
        """

        # Step counter (one step = n_points cost evaluations)
        self.stp = 0

        # Path
        self.path = self.base_path+"/"+str(run)

    def sample(self):
        """
        Samples a new point from the agent's distribution.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def step(self, c):
        """
        Performs one optimization step.

        Args:
            c: The cost value(s) of the sampled point(s).

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def render(self):
        """
        Renders the agent's current state.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def summary(self) -> None:
        """
        Prints a summary of the agent's configuration.
        """

        spacer("Using "+self.name+" algorithm with "+str(self.n_points)+" points")
        spacer("Problem dimensionality is "+str(self.dim))

    def ndof(self) -> int:
        """
        Returns the number of degrees of freedom of the agent.
        """

        return self.n_points

    def done(self) -> bool:
        """
        Checks if the agent has reached its termination condition.

        Returns:
            True if the maximum number of steps has been reached, False otherwise.
        """

        if (self.stp == self.n_steps_max): return True
        return False
