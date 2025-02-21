# Custom imports
from sparkle.src.trainer.base  import base_trainer
from sparkle.src.agent.agent   import agent_factory
from sparkle.src.utils.timer   import timer
from sparkle.src.env.parallel  import parallel
from sparkle.src.utils.default import set_default
from sparkle.src.utils.error   import error

###############################################
### Class for regular trainer
class regular(base_trainer):
    def __init__(self, path, pms):

        # Set parameters
        self.base_path    = path
        self.render_every = set_default("render_every", 100000, pms)

        # Initialize environment
        self.env = parallel.environments(path, pms.environment)

        # Initialize agent
        self.agent = agent_factory.create(pms.agent.name,
                                          path   = path,
                                          spaces = self.env.spaces,
                                          pms    = pms.agent)

        # Check compatibility between the number of parallel workers
        # and the number of degrees of freedom required by the agent
        if (self.agent.ndof()%parallel.size !=0):
            error("trainer::regular", "init", "Number of degress of freedom of the agent must be a multiple of the number of parallel workers", call_exit=False)
            self.env.close()
            parallel.finalize()

        # Initialize timer
        self.timer_global = timer("global   ")

    # Reset
    def reset(self, run):

        super().reset(run)
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

            # Compute cost
            c = self.env.cost(x)

            # Store data
            self.store_data(x, c)

            # Render if necessary
            if (self.it%self.render_every == 0):
                self.env.render(x, c)

            # Step and output
            self.agent.step(x, c)

            # Print
            self.print()

            self.it += 1

        # Dump stored data to file
        self.dump_data()

        # Close timer and show
        self.timer_global.toc()
        self.timer_global.show()


