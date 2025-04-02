import os
from types import SimpleNamespace

from numpy import ndarray

from sparkle.src.agent.agent import agent_factory
from sparkle.src.env.parallel import parallel
from sparkle.src.plot.plot import render_1D_regular, render_2D_regular
from sparkle.src.trainer.base import BaseTrainer
from sparkle.src.utils.default import set_default
from sparkle.src.utils.error import error
from sparkle.src.utils.timer import Timer


###############################################
### Class for regular trainer
class Regular(BaseTrainer):
    def __init__(self, path: str, pms: SimpleNamespace) -> None:

        # Set parameters
        self.base_path    = path
        self.render_every = set_default("render_every", 100000, pms)
        self.renderer     = set_default("renderer", "env", pms)

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
            error("trainer::regular", "init",
                  "Number of degress of freedom of the agent must be a multiple of the number of parallel workers",
                  call_exit=False)
            self.env.close()
            parallel.finalize()

        # Initialize timer
        self.timer_global = Timer("global   ")

    # Reset
    def reset(self, run: int) -> None:

        super().reset(run)
        self.env.reset(run)
        self.agent.reset(run)

    # Optimize
    def optimize(self) -> None:

        self.timer_global.tic()
        self.it = 0

        # Loop until done
        while (not self.agent.done()):

            x = self.agent.sample()
            c = self.env.cost(x)

            self.store_data(x, c)
            self.render(x, c)
            self.agent.step(x, c)
            self.print()

            self.it += 1

        self.dump_data()

        self.timer_global.toc()
        self.timer_global.show()

    # Handle rendering
    def render(self, x: ndarray, c: ndarray) -> None:

        if (self.it%self.render_every != 0): return

        if (self.renderer == "env"): self.env.render(x, c)
        if (self.renderer == "trainer"):

            # Check dimension
            if (self.env.spaces.dim > 2):
                error("trainer::regular", "render",
                      "trainer rendering is only available for dim <= 2")

            # Create output folder
            if (self.it_plt == 0): os.makedirs(self.path+'/png', exist_ok=True)
            filename = self.path+"/png/"+str(self.it_plt)+".png"

            # Render depending on dimension
            if (self.env.spaces.dim == 1):

                # Generate cost map once at first call to save computational time
                if not hasattr(self, "cost_map"):
                    self.x_plot, self.cost_map = self.env.generate_cost_map_1D()

                render_1D_regular(filename, self.env.spaces, x, c,
                                  self.x_plot, self.cost_map)

            # Render depending on dimension
            if (self.env.spaces.dim == 2):

                # Generate cost map once at first call to save computational time
                if not hasattr(self, "cost_map"):
                    self.x_plot, self.y_plot, self.cost_map = self.env.generate_cost_map_2D()

                render_2D_regular(filename, self.env.spaces, x, c,
                                  self.x_plot, self.y_plot, self.cost_map)

        self.it_plt += 1
