# Custom imports
from sparkle.src.trainer.base     import *
from sparkle.src.env.environments import *
from sparkle.src.agent.agent      import *

###############################################
### Class for regular trainer
class regular(base_trainer):
    def __init__(self, env_pms, agent_pms, path, pms):

        # Initialize environment
        self.env = environments(path, env_pms)

        # Initialize from input
        self.dim = self.env.dim()

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.name,
                                          dim = self.dim,
                                          pms = agent_pms)

        # Check compatibility between the number of parallel workers
        # and the number of degrees of freedom required by the agent
        if (self.agent.ndof()%mpi.size !=0):
            error("trainer::regaular", "init", "Number of degress of freedom of the agent must be a multiple of the number of parallel workers")

        # Initialize timer
        self.timer_global = timer("global   ")

    # Optimize
    def optimize(self, path, run):

        # Start global timer
        self.timer_global.tic()

        # Reset agent
        agent.reset()

        # Loop until done
        while (not self.agent.done()):

            x = agent.dof()
            c = env.cost(x)
            agent.step(c)

        # Close timer and show
        self.timer_global.toc()
        self.timer_global.show()
