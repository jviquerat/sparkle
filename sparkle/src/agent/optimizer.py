import numpy as np

from sparkle.src.agent.cem import CEM
from sparkle.src.agent.cmaes import CMAES
from sparkle.src.agent.pbo import PBO
from sparkle.src.agent.pso import PSO

# Local factory
from sparkle.src.core.factory import Factory

local_factory = Factory()

local_factory.register("pso",   PSO)
local_factory.register("cmaes", CMAES)
local_factory.register("cem",   CEM)
local_factory.register("pbo",   PBO)

###############################################

class Optimizer():
    """
    A transparent class for calling optimizers with a simple cost function.

    This class provides a simplified interface for using various optimization
    algorithms without the need to define a full environment. It is designed
    for sequential execution and does not adhere to the standard agent
    interface used elsewhere in the framework.
    """
    def __init__(self, name, spaces, pms, cost):
        """
        Initializes the Optimizer.

        Args:
            name: The name of the optimization algorithm to use (e.g., "pso", "cmaes").
            spaces: The search space definition.
            pms: A SimpleNamespace object containing parameters for the optimizer.
            cost: The cost function to be minimized.
        """

        # Initialize agent
        self.agent = local_factory.create(name, path=".", spaces=spaces, pms=pms)

        # Cost function
        self.cost = cost

        # Initial reset
        self.reset(0)

    def reset(self, run):
        """
        Resets the optimizer for a new run.

        Args:
            run: The run number.
        """

        self.agent.reset(run)

    def optimize(self):
        """
        Performs the optimization process.

        This method iteratively samples points, evaluates the cost function,
        and updates the optimizer until the termination condition is met.

        Returns:
            A tuple containing:
                - The best point found (NumPy array).
                - The best score achieved (float).
        """

        # Loop until done
        best_x     = None
        best_score = 1.0e15
        while (not self.agent.done()):

            x = self.agent.sample()

            n_pts = x.shape[0]
            c     = np.zeros(n_pts)
            for p in range(n_pts):
                res = self.cost(x[p])
                if (hasattr(res, "__len__")):
                    c[p] = res[0]
                else:
                    c[p] = res

                if (c[p] < best_score):
                    best_score = c[p]
                    best_x     = x[p]

            self.agent.step(x, c)

        return best_x, best_score
