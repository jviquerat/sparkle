# Generic imports
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy             as np

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
        self.timer_global = timer("global   ")

    # Reset
    def reset(self, run):

        super().reset(run)
        self.env.reset(run)
        self.agent.reset(run)

    # Optimize
    def optimize(self):

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
                ax  = fig.add_subplot(111)

                ax.set_xlim([self.env.spaces.xmin[0], self.env.spaces.xmax[0]])
                ax.set_ylim([self.env.spaces.vmin,    self.env.spaces.vmax])
                ax.grid()
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.plot(self.x_plot, self.cost_map, label="f(x)")
                ax.set_ylabel('y')

                ax.scatter(x[:,0], c[:], c="black", marker='o', alpha=0.8, label="samples")
                ax.legend(loc='upper left')

                filename = self.path+"/png/"+str(self.it_plt)+".png"
                fig.tight_layout()
                plt.savefig(filename, dpi=100)
                plt.close()

            # Render depending on dimension
            if (self.env.spaces.dim == 2):

                plt.clf()
                fig = plt.figure()
                ax  = fig.add_subplot(111)
                fig.set_size_inches(3, 3)
                fig.subplots_adjust(0,0,1,1)

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

                filename = self.path+"/png/"+str(self.it_plt)+".png"
                plt.savefig(filename, dpi=100)
                plt.close()

        self.it_plt += 1
