# Generic imports
import os
import sys
import time
import shutil
import numpy as np

# Custom imports
from sparkle.src.env.parallel    import parallel
from sparkle.src.pex.pex         import pex_factory
from sparkle.src.model.model     import model_factory
from sparkle.src.plot.plot       import render_1D_metamodel, render_2D_metamodel
from sparkle.src.utils.prints    import liner
from sparkle.src.utils.error     import error

# Create pex, compute costs and generate model
def generate(env_pms, pex_pms, model_pms, path):

    # Initialize environment
    env = parallel.environments(path, env_pms)

    # Initialize pex
    pex = pex_factory.create(pex_pms.name,
                             spaces = env.spaces,
                             pms    = pex_pms)

    # Initialize model
    model = model_factory.create(model_pms.name,
                                 spaces = env.spaces,
                                 pms    = model_pms)

    # Compute costs
    pex_costs = env.evaluate(pex.x)

    # Initialize model
    model.build(pex.x, pex_costs)
    model.dump(path+"/model.dat")

    # Render
    if (env.spaces.dim > 2):
        error("model::generator", "generate",
              "rendering is only available for dim <= 2")

    if (env.spaces.dim == 1):
        x_plot, cost_map = env.generate_cost_map_1D()

        xx          = np.reshape(x_plot, (-1,1))
        y_mu, y_std = model.evaluate(xx)

        filename = path+"/model.png"
        fct_name = "std"
        render_1D_metamodel(filename, env.spaces, pex.x, pex_costs,
                            x_plot, cost_map,
                            y_mu, y_std, y_std, fct_name,
                            highlight_last=False)

    if (env.spaces.dim == 2):
        x_plot, y_plot, cost_map = env.generate_cost_map_2D()

        n_plot  = cost_map.shape[0]
        y_mu    = np.zeros((n_plot, n_plot))
        y_std   = np.zeros((n_plot, n_plot))

        for i in range(n_plot):
            for j in range(n_plot):
                xx         = np.array([[x_plot[i,j], y_plot[i,j]]])
                mu, std    = model.evaluate(xx)
                y_mu[i,j]  = mu[0]
                y_std[i,j] = std[0]

        filename = path+"/model.png"
        fct_name = "std"
        render_2D_metamodel(filename, env.spaces, pex.x, pex_costs,
                            x_plot, y_plot, cost_map,
                            y_mu, y_std, y_std, fct_name,
                            highlight_last=False)

    # Close environments
    env.close()
