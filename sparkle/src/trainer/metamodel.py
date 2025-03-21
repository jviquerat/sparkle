# Generic imports
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Custom imports
from sparkle.src.trainer.base  import base_trainer
from sparkle.src.agent.agent   import agent_factory
from sparkle.src.utils.timer   import timer
from sparkle.src.env.parallel  import parallel
from sparkle.src.pex.pex       import pex_factory
from sparkle.src.model.model   import model_factory
from sparkle.src.utils.default import set_default
from sparkle.src.plot.plot     import render_1D_metamodel, render_2D_metamodel
from sparkle.src.utils.error   import error, warning
from sparkle.src.utils.prints  import spacer

###############################################
### Class for metamodel-based trainer
class metamodel(base_trainer):
    def __init__(self, path, pms):

        # Set parameters
        self.base_path    = path
        self.render_every = set_default("render_every", 100000, pms)
        self.renderer     = set_default("renderer", "env", pms)

        # Initialize environment
        self.env = parallel.environments(path, pms.environment)

        # Initialize pex
        self.pex = pex_factory.create(pms.pex.name,
                                      spaces = self.env.spaces,
                                      pms    = pms.pex)

        # Initialize model
        self.model = model_factory.create(pms.model.name,
                                          spaces = self.env.spaces,
                                          path   = path,
                                          pms    = pms.model)

        # Initialize agent
        self.agent = agent_factory.create(pms.agent.name,
                                          path   = path,
                                          spaces = self.env.spaces,
                                          model  = self.model,
                                          pms    = pms.agent)

        # Initialize timer
        self.timer_global  = timer("global  ")
        self.timer_opt     = timer("opt     ")
        self.timer_mod     = timer("model   ")
        self.timer_pex     = timer("pex ")

    # Reset
    def reset(self, run):

        super().reset(run)
        self.env.reset(run)
        self.pex.reset()
        self.model.reset()
        self.agent.reset(run)

    # Optimize
    def optimize(self):

        self.timer_global.tic()

        # Check if model is loaded from file or if it must be computed
        if (self.model.load_model_):
            self.model.load()
            self.store_data(self.model.x, self.model.y)

            # Keep local copy
            self.x = self.model.x
            self.y = self.model.y

            spacer("Loaded model")
            self.initial_print()
        else:
            self.timer_pex.tic()

            pex_costs = self.env.evaluate(self.pex.x)
            self.store_data(self.pex.x, pex_costs)

            self.timer_pex.toc()
            self.timer_pex.show()

            # Build and dump model
            self.timer_mod.tic()
            self.model.build(self.pex.x, pex_costs)
            self.model.dump()

            # Keep local copy
            self.x = self.pex.x
            self.y = pex_costs

            spacer("Built initial model")
            self.initial_print()

            self.timer_mod.toc()
            self.timer_mod.show()

        # Such agents are only sequential for now
        if (parallel.size > 1):
            warning("trainer::metamodel", "optimize",
                    "only samples generation can be performed in parallel")
            return

        # Set counter
        self.timer_opt.tic()
        self.it = 0

        # Loop until done
        while (not self.agent.done()):

            # Sample new point
            x = self.agent.sample()

            # Compute cost
            c = self.env.cost(x)

            # Store data
            self.store_data(x, c)

            # Update local copy
            self.x = np.vstack((self.x, x))
            self.y = np.hstack((self.y, c))

            # Render
            self.render(self.x, self.y)

            # Update model
            self.model.build(self.x, self.y)

            # Step and output
            self.agent.step(x, c)
            self.print()

            self.it += 1

        # Dump stored data to file
        self.dump_data()

        # Close timers
        self.timer_opt.toc()
        self.timer_global.toc()
        self.timer_opt.show()
        self.timer_global.show()

    # Handle rendering
    def render(self, x, c):

        if (self.it%self.render_every != 0): return

        if (self.renderer == "env"): self.env.render(x, c)
        if (self.renderer == "trainer"):

            # Check dimension
            if (self.env.spaces.dim > 2):
                error("trainer::metamodel", "render",
                      "trainer rendering is only available for dim <= 2")

            # Create output folder
            if (self.it_plt == 0): os.makedirs(self.path+'/png', exist_ok=True)
            filename = self.path+"/png/"+str(self.it_plt)+".png"

            # Render depending on dimension
            if (self.env.spaces.dim == 1):

                # Generate cost map once at first call to save computational time
                if not hasattr(self, "cost_map"):
                    self.x_plot, self.cost_map = self.env.generate_cost_map_1D()

                xx          = np.reshape(self.x_plot, (-1,1))
                y_mu, y_std = self.model.evaluate(xx)
                infill      = self.agent.infill(xx)

                fct_name = "infill"
                render_1D_metamodel(filename, self.env.spaces, x, c,
                                    self.x_plot, self.cost_map,
                                    y_mu, y_std, infill, fct_name)

            # Render depending on dimension
            if (self.env.spaces.dim == 2):

                # Generate cost map once at first call to save computational time
                if not hasattr(self, "cost_map"):
                    self.x_plot, self.y_plot, self.cost_map = self.env.generate_cost_map_2D()
                    self.xx = np.column_stack([self.x_plot.ravel(), self.y_plot.ravel()])

                n_plot  = self.cost_map.shape[0]
                mu, std = self.model.evaluate(self.xx)
                infill = self.agent.infill(self.xx)
                y_mu    = np.reshape(mu, (n_plot, n_plot))
                y_std   = np.reshape(std, (n_plot, n_plot))
                infill  = np.reshape(infill, (n_plot, n_plot))

                fct_name = "acquisition function"
                render_2D_metamodel(filename, self.env.spaces, x, c,
                                    self.x_plot, self.y_plot, self.cost_map,
                                    y_mu, y_std, infill, fct_name)

        self.it_plt += 1

    # Print after building or loading model
    def initial_print(self):

        gs     = f"{self.best_score:.5e}"
        gb     = np.array2string(self.best_x, precision=5,
                                 floatmode='fixed', threshold=4, separator=',')

        spacer("Best initial score = "+str(gs)+" for x = "+str(gb))
