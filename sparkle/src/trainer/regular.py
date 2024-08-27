# Custom imports
from sparkle.src.trainer.base import base_trainer
from sparkle.src.agent.agent  import agent_factory
from sparkle.src.utils.timer  import timer
from sparkle.src.env.parallel import parallel

###############################################
### Class for regular trainer
class regular(base_trainer):
    def __init__(self, env_pms, agent_pms, path, pms):

        # Set parameters
        self.render_every = 100000
        if hasattr(pms, "render_every"): self.render_every = pms.render_every

        # Initialize environment
        self.env = parallel.environments(path, env_pms)

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.name,
                                          path = path,
                                          dim  = self.env.dim(),
                                          x0   = self.env.x0(),
                                          xmin = self.env.xmin(),
                                          xmax = self.env.xmax(),
                                          pms  = agent_pms)

        # Check compatibility between the number of parallel workers
        # and the number of degrees of freedom required by the agent
        if (self.agent.ndof()%parallel.size() !=0):
            error("trainer::regular", "init", "Number of degress of freedom of the agent must be a multiple of the number of parallel workers")

        # Initialize timer
        self.timer_global = timer("global   ")

    # Reset
    def reset(self, run):

        self.env.reset(run)
        self.agent.reset(run)

    # Optimize
    def optimize(self):

        # Set counter
        self.timer_global.tic()
        self.it = 0

        # Loop until done
        while (not self.agent.done()):

            # Sample points
            x = self.agent.sample()

            # Render if necessary
            if (self.it%self.render_every == 0):
                self.env.render(x)

            # Compute cost and step
            c = self.env.cost(x)
            self.agent.step(x, c)
            self.agent.print()

            self.it += 1

        self.agent.dump()

        # Close timer and show
        self.timer_global.toc()
        self.timer_global.show()


