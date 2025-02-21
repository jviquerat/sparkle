# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.trainer.base  import base_trainer
from sparkle.src.agent.agent   import agent_factory
from sparkle.src.utils.timer   import timer
from sparkle.src.env.parallel  import parallel
from sparkle.src.pex.pex       import pex_factory
from sparkle.src.utils.default import set_default
from sparkle.src.utils.error   import error, warning

###############################################
### Class for metamodel-based trainer
class metamodel(base_trainer):
    def __init__(self, path, pms):

        # Set parameters
        self.base_path      = path
        self.render_every   = set_default("render_every", 100000, pms)
        self.plot_estimates = set_default("plot_estimates", False, pms)

        # Initialize environment
        self.env = parallel.environments(path, pms.environment)

        # Initialize pex
        self.pex = pex_factory.create(pms.pex.name,
                                      spaces = self.env.spaces,
                                      pms    = pms.pex)

        # Initialize agent
        self.agent = agent_factory.create(pms.agent.name,
                                          path   = path,
                                          spaces = self.env.spaces,
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
        self.agent.reset(run)

    # Optimize
    def optimize(self):

        self.timer_global.tic()

        # Check if model is loaded from file or if it must be computed
        if (self.agent.load_model_):
            self.agent.load_model()
            self.store_data(self.agent.x_, self.agent.y_)
        else:
            self.timer_pex.tic()

            if (self.pex.n_points%parallel.size != 0):
                error("trainer::metamodel", "optimize",
                      "nb of pex points should be a multiple of nb of parallel envs")

            n_steps   = self.pex.n_points//parallel.size
            pex_costs = np.zeros(self.pex.n_points)

            step = 0
            while (step < n_steps):
                end = "\r"
                if (step == n_steps-1): end = "\n"
                i_start = step*parallel.size
                i_end   = (step+1)*parallel.size - 1
                print("# Computing pex individuals #"+str(i_start)+" to #"+str(i_end), end=end)

                xp = np.zeros((parallel.size, self.env.spaces.dim))
                for k in range(parallel.size):
                    xp[k,:] = self.pex.point(step*parallel.size + k)

                c = self.env.cost(xp)
                for k in range(parallel.size):
                    pex_costs[step*parallel.size + k] = c[k]

                self.store_data(xp, c)
                step += 1

            self.timer_pex.toc()
            self.timer_pex.show()

            # Build and dump model
            self.timer_mod.tic()
            self.agent.build_initial_model(self.pex.x, pex_costs)
            self.agent.dump_model(self.agent.path+"/model.dat")
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

            # Render if necessary
            if (self.it%self.render_every == 0):
                self.render(x, c)

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

    # Rendering interface to output plots with metamodel informations
    # x_last is the unevaluated last sample point
    def render(self, x_last, c_last):

        # Rendering with metamodel informations (can be expensive to compute)
        if (self.plot_estimates):

            if (self.env.spaces.dim > 2):
                error("metamodel", "render",
                      "plot_estimates is only available for dim <= 2")

            xmin    = self.env.spaces.xmin
            xmax    = self.env.spaces.xmax
            x_last  = np.reshape(x_last, (-1))

            if (self.env.spaces.dim == 1):
                nx_plot = 200
                x_plot  = np.linspace(xmin[0], xmax[0], num=nx_plot)
                x_plot  = np.reshape(x_plot, (-1,1))

                x0      = x_plot
                y       = self.agent.model.evaluate(x0)
                ei      = self.agent.exp_imp(x0)
                y_mu    = y[0]
                y_std   = y[1]

                pms       = types.SimpleNamespace()
                pms.x_mu  = x_plot.squeeze()
                pms.y_mu  = y_mu
                pms.y_std = y_std
                pms.ei    = ei
                pms.x_ei  = x_last

                self.env.render(self.agent.x_, c_last, pms=pms)

            if (self.env.spaces.dim == 2):
                nx = 100
                ny = 100

                x     = np.linspace(xmin[0], xmax[0], num=nx)
                y     = np.linspace(xmax[1], xmin[1], num=ny)
                x, y  = np.array(np.meshgrid(x, y))
                x0    = np.zeros((nx,ny))
                y_mu  = np.zeros((nx,ny))
                y_std = np.zeros((nx,ny))

                for i in range(nx):
                    for j in range(ny):
                        xx = np.array((x[i,j],y[i,j]))
                        xx = np.reshape(xx, (-1,2))
                        yy = self.agent.model.evaluate(xx)

                        y_mu[i,j]  = yy[0]
                        y_std[i,j] = yy[1]

                pms       = types.SimpleNamespace()
                pms.y_mu  = y_mu
                pms.y_std = y_std
                pms.x_ei  = x_last

                self.env.render(self.agent.x_, c_last, pms=pms)

        else:
            # Regular rendering without metamodel informations
            x = self.agent.denormalize(x_last)
            self.env.render(x, c_last)
