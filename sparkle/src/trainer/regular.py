# Custom imports
from sparkle.src.trainer.base     import *
from sparkle.src.env.environments import *
from sparkle.src.agent.agent      import *

###############################################
### Class for regular trainer
class regular(base_trainer):
    def __init__(self, env_pms, agent_pms, path, pms):

        # Set parameters
        self.render_every = 100000
        if hasattr(pms, "render_every"): self.render_every = pms.render_every

        # Initialize environment
        self.env = environments(path, env_pms)

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.name,
                                          dim  = self.env.dim(),
                                          xmin = self.env.xmin(),
                                          xmax = self.env.xmax(),
                                          pms  = agent_pms)

        # Check compatibility between the number of parallel workers
        # and the number of degrees of freedom required by the agent
        if (self.agent.ndof()%mpi.size !=0):
            error("trainer::regaular", "init", "Number of degress of freedom of the agent must be a multiple of the number of parallel workers")

        # Initialize timer
        self.timer_global = timer("global   ")

    # Optimize
    def optimize(self, run):

        # Start global timer
        self.timer_global.tic()

        # Reset agent
        self.agent.reset()

        # Set counter
        self.it = 0

        # Loop until done
        while (not self.agent.done()):

            x = self.agent.dof()
            c = self.env.cost(x)
            self.agent.step(c)
            self.agent.print()

            if (self.it%self.render_every == 0):
                self.env.render(self.agent.dof())

            self.it += 1

        # Close timer and show
        self.timer_global.toc()
        self.timer_global.show()


