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

            spacer()
            print("Loaded model")
            self.initial_print()
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
            self.model.build(self.pex.x, pex_costs)
            self.model.dump(self.agent.path+"/model.dat")

            # Keep local copy
            self.x = self.pex.x
            self.y = pex_costs

            spacer()
            print("Built initial model")
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
                error("trainer::regularl", "render",
                      "trainer rendering is only available for dim <= 2")

            # Generate cost map once at first call to save computational time
            if not hasattr(self, "cost_map"): self.generate_cost_map()

            # Create output folder
            if (self.it_plt == 0): os.makedirs(self.path+'/png', exist_ok=True)

            # Render depending on dimension
            if (self.env.spaces.dim == 1):

                plt.clf()
                fig = plt.figure()

                xx          = np.reshape(self.x_plot, (-1,1))
                y_mu, y_std = self.model.evaluate(xx)
                exp_imp     = self.agent.exp_imp(xx)

                ax = fig.add_subplot(211)
                ax.set_xlim([self.env.spaces.xmin[0], self.env.spaces.xmax[0]])
                ax.set_ylim([self.env.spaces.vmin,    self.env.spaces.vmax])
                ax.grid()
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.plot(self.x_plot, self.cost_map, label="f(x)")
                ax.set_ylabel('y')

                ax.scatter(x[:,0], c[:], c="black", marker='o', alpha=0.8, label="samples")
                ax.scatter(x[-1,0], c[-1], c='red', marker='o', alpha=0.8)
                ax.legend(loc='upper left')

                ax.plot(self.x_plot, y_mu, linestyle='dashed', label="model")
                ax.fill_between(self.x_plot, y_mu-y_std, y_mu+y_std, alpha=0.2,
                                label="confidence interval")

                ax = fig.add_subplot(212)
                ax.set_xlim([self.env.spaces.xmin[0], self.env.spaces.xmax[0]])
                ax.plot(self.x_plot, -exp_imp, color='r')
                ax.grid()
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_ylabel('expected improvement')

                filename = self.path+"/png/"+str(self.it_plt)+".png"
                fig.tight_layout()
                plt.savefig(filename, dpi=100)
                plt.close()

            # Render depending on dimension
            if (self.env.spaces.dim == 2):

                plt.clf()
                fig = plt.figure()
                fig.set_size_inches(8, 3)
                fig.tight_layout()
                plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
                plt.rcParams.update({'axes.titlesize': 'small'})

                y_mu    = np.zeros((self.n_plot, self.n_plot))
                y_std   = np.zeros((self.n_plot, self.n_plot))
                exp_imp = np.zeros((self.n_plot, self.n_plot))

                for i in range(self.n_plot):
                    for j in range(self.n_plot):
                        xx = np.array([[self.x_plot[i,j], self.y_plot[i,j]]])
                        y_mu[i,j], y_std[i,j] = self.model.evaluate(xx)
                        exp_imp[i,j]          = self.agent.exp_imp(xx)

                ax = fig.add_subplot(131)
                ax.set_xlim([self.env.spaces.xmin[0], self.env.spaces.xmax[0]])
                ax.set_ylim([self.env.spaces.xmin[1], self.env.spaces.xmax[1]])
                ax.axis('off')
                ax.imshow(self.cost_map,
                          extent=[self.env.spaces.xmin[0], self.env.spaces.xmax[0],
                                  self.env.spaces.xmin[1], self.env.spaces.xmax[1]],
                          vmin=self.env.spaces.vmin, vmax=self.env.spaces.vmax, alpha=0.8, cmap='RdBu_r')

                cnt = ax.contour(self.x_plot, self.y_plot, self.cost_map,
                                 levels=self.env.spaces.levels, colors='black', alpha=0.5)
                ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
                ax.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)
                ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.8)
                ax.set_title("f(x)")

                ax = fig.add_subplot(132)
                ax.axis('off')
                ax.imshow(y_mu,
                          extent=[self.env.spaces.xmin[0], self.env.spaces.xmax[0],
                                  self.env.spaces.xmin[1], self.env.spaces.xmax[1]],
                          vmin=self.env.spaces.vmin, vmax=self.env.spaces.vmax, alpha=0.8, cmap='RdBu_r')
                cnt = ax.contour(self.x_plot, self.y_plot, y_mu,
                                 levels=self.env.spaces.levels, colors='black', alpha=0.5)
                ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
                ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.8)
                ax.set_title("model")

                ax = fig.add_subplot(133)
                ax.axis('off')
                ax.imshow(y_std,
                          extent=[self.env.spaces.xmin[0], self.env.spaces.xmax[0],
                                  self.env.spaces.xmin[1], self.env.spaces.xmax[1]],
                          alpha=0.8, cmap='RdBu_r')
                ax.set_title("confidence interval")

                filename = self.path+"/png/"+str(self.it_plt)+".png"
                plt.savefig(filename, dpi=100)
                plt.close()

        self.it_plt += 1




















        # Rendering with metamodel informations (can be expensive to compute)
        # if (self.plot_estimates):

        #     if (self.env.spaces.dim > 2):
        #         error("metamodel", "render",
        #               "plot_estimates is only available for dim <= 2")

        #     xmin    = self.env.spaces.xmin
        #     xmax    = self.env.spaces.xmax
        #     x_last  = np.reshape(x_last, (-1))

        #     if (self.env.spaces.dim == 1):
        #         nx_plot = 200
        #         x_plot  = np.linspace(xmin[0], xmax[0], num=nx_plot)
        #         x_plot  = np.reshape(x_plot, (-1,1))

        #         x0      = x_plot
        #         y       = self.agent.model.evaluate(x0)
        #         ei      = self.agent.exp_imp(x0)
        #         y_mu    = y[0]
        #         y_std   = y[1]

        #         pms       = types.SimpleNamespace()
        #         pms.x_mu  = x_plot.squeeze()
        #         pms.y_mu  = y_mu
        #         pms.y_std = y_std
        #         pms.ei    = ei
        #         pms.x_ei  = x_last

        #         self.env.render(self.model.x, c_last, pms=pms)

        #     if (self.env.spaces.dim == 2):
        #         nx = 100
        #         ny = 100

        #         x     = np.linspace(xmin[0], xmax[0], num=nx)
        #         y     = np.linspace(xmax[1], xmin[1], num=ny)
        #         x, y  = np.array(np.meshgrid(x, y))
        #         x0    = np.zeros((nx,ny))
        #         y_mu  = np.zeros((nx,ny))
        #         y_std = np.zeros((nx,ny))

        #         for i in range(nx):
        #             for j in range(ny):
        #                 xx = np.array((x[i,j],y[i,j]))
        #                 xx = np.reshape(xx, (-1,2))
        #                 yy = self.agent.model.evaluate(xx)

        #                 y_mu[i,j]  = yy[0]
        #                 y_std[i,j] = yy[1]

        #         pms       = types.SimpleNamespace()
        #         pms.y_mu  = y_mu
        #         pms.y_std = y_std
        #         pms.x_ei  = x_last

        #         self.env.render(self.model.x, c_last, pms=pms)

        # else:
        #     Regular rendering without metamodel informations
        #     self.env.render(x_last, c_last)

    # Print after building or loading model
    def initial_print(self):

        gs     = f"{self.best_score:.5e}"
        gb     = np.array2string(self.best_x, precision=5,
                                 floatmode='fixed', threshold=4, separator=',')

        spacer()
        print("Best initial score = "+str(gs)+" for x = "+str(gb))
