import numpy as np

from sparkle.src.env.parallel import parallel
from sparkle.src.model.model import model_factory
from sparkle.src.pex.pex import pex_factory
from sparkle.src.plot.plot import render_1D_metamodel, render_2D_metamodel
from sparkle.src.utils.error import error
from sparkle.src.utils.timer import Timer


def generate(env_pms, pex_pms, model_pms, path):
    """
    Generates a surrogate model and visualizes it.

    This function initializes the environment, a point exploration (Pex)
    algorithm, and a surrogate model. It then evaluates the cost of the
    points sampled by the Pex algorithm, builds the surrogate model,
    and visualizes the model if the dimensionality is 1 or 2.

    Args:
        env_pms: Parameters for the environment.
        pex_pms: Parameters for the Pex algorithm.
        model_pms: Parameters for the surrogate model.
        path: The base path for storing results.
    """

    # Initialize environment
    env = parallel.environments(path, env_pms)

    # Initialize pex
    timer_pex = Timer("pex generation ")
    timer_pex.tic()
    pex = pex_factory.create(pex_pms.name,
                             spaces = env.spaces,
                             pms    = pex_pms)
    pex.summary()
    timer_pex.toc()
    timer_pex.show()

    # Evaluate costs
    timer_eval = Timer("pex evaluation ")
    timer_eval.tic()
    pex_costs = env.evaluate(pex.x)
    timer_eval.toc()
    timer_eval.show()

    # Initialize model
    model = model_factory.create(model_pms.name,
                                 spaces = env.spaces,
                                 path   = path,
                                 pms    = model_pms)

    # Initialize model
    timer_model = Timer("model generation")
    timer_model.tic()
    model.build(pex.x, pex_costs)
    timer_model.toc()
    timer_model.show()
    model.dump()

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
                            n_new=0)

    if (env.spaces.dim == 2):
        x_plot, y_plot, cost_map = env.generate_cost_map_2D()
        xx = np.column_stack([x_plot.ravel(), y_plot.ravel()])

        n_plot  = cost_map.shape[0]
        mu, std = model.evaluate(xx)
        y_mu    = np.reshape(mu, (n_plot, n_plot))
        y_std   = np.reshape(std, (n_plot, n_plot))

        filename = path+"/model.png"
        fct_name = "std"
        render_2D_metamodel(filename, env.spaces, pex.x, pex_costs,
                            x_plot, y_plot, cost_map,
                            y_mu, y_std, y_std, fct_name,
                            n_new=0)

    env.close()
