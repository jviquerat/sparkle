import types

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.diff_pex import DiffPex
from sparkle.src.utils.seeds import set_seeds


###############################################
def test_differentiable_maximin():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms            = types.SimpleNamespace()
    pms.n_points   = n_points
    pms.lr         = 1.0
    pms.n_steps    = 50
    pms.cost_name  = "maximin"

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)

    # Instantiate directly to test logic
    pex = DiffPex(s, pms)

    # Check number of points
    assert pex.n_points == n_points

    # Check bounds
    assert np.all(pex.x >= xmin)
    assert np.all(pex.x <= xmax)

    # Check spacing (points should be reasonably distinct)
    dists = np.linalg.norm(pex.x[:, None, :] - pex.x[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    # With 10 points in unit square, sqrt(1/10) ~ 0.3
    assert min_dist > 0.1

def test_differentiable_minimax():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms            = types.SimpleNamespace()
    pms.n_points   = n_points
    pms.lr         = 1.0
    pms.n_steps    = 50
    pms.cost_name  = "minimax"

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)

    # Instantiate directly to test logic
    pex = DiffPex(s, pms)

    # Check number of points
    assert pex.n_points == n_points

    # Check bounds
    assert np.all(pex.x >= xmin)
    assert np.all(pex.x <= xmax)

    # We expect Minimax to also distribute points reasonably well
    dists = np.linalg.norm(pex.x[:, None, :] - pex.x[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    assert min_dist > 0.1

def test_differentiable_maxpro():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 10

    pms            = types.SimpleNamespace()
    pms.n_points   = n_points
    pms.cost       = "maxpro"
    pms.lr         = 1.0
    pms.n_steps    = 50

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)

    # Instantiate directly to test logic
    pex = DiffPex(s, pms)

    # Check number of points
    assert pex.n_points == n_points

    # Check bounds
    assert np.all(pex.x >= xmin)
    assert np.all(pex.x <= xmax)

