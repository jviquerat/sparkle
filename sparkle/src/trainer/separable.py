# Generic imports
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.trainer.base    import BaseTrainer
from sparkle.src.agent.agent     import agent_factory
from sparkle.src.utils.timer     import Timer
from sparkle.src.env.parallel    import parallel
from sparkle.src.utils.error     import error
from sparkle.src.pex.pex         import pex_factory
from sparkle.src.model.sepnet    import SepNet

###############################################
### Class for separable trainer
class Separable(BaseTrainer):
    def __init__(self, env_pms, agent_pms, path, pms):

        # Set parameters
        self.r_dim = pms.r_dim
        self.render_every = 100000
        if hasattr(pms, "render_every"): self.render_every = pms.render_every

        # Initialize environment
        self.env = parallel.environments(path, env_pms)

        # Initialize pex
        self.pex = pex_factory.create(pms.pex.name,
                                      spaces = self.env.spaces,
                                      pms    = pms.pex)

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.name,
                                          path   = path,
                                          spaces = self.env.spaces,
                                          pms    = agent_pms)

        self.model = SepNet(r_dim   = self.r_dim,
                            spaces  = self.env.spaces,
                            pms_opt = pms.opt)

        # Initialize timer
        self.timer_global = Timer("global   ")

    # Reset
    def reset(self, run):

        self.env.reset(run)
        self.pex.reset()
        self.model.reset()

    # Optimize
    def optimize(self):

        costs = np.zeros(self.pex.n_points())
        for i in range(self.pex.n_points()):
            costs[i] = self.env.cost(self.pex.point(i))

        self.model.build(self.pex.x(), costs)

        self.model.plot_model_2d(self.pex.x(), self.env.cost)

        exit()
