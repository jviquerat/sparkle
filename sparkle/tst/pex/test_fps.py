# Generic imports
import types
import numpy as np

# Custom imports
from sparkle.src.pex.fps     import FPS
from sparkle.src.pex.random  import Random
from sparkle.src.env.spaces  import EnvSpaces
from sparkle.src.utils.seeds import set_seeds

###############################################
### Test furthest point sampling
def test_fps():

    set_seeds(0)

    dim      = 2
    xmin     = np.array([0.0, 0.0])
    xmax     = np.array([1.0, 1.0])
    n_points = 100

    pms          = types.SimpleNamespace()
    pms.n_points = n_points

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    s = EnvSpaces(loc_space)
    pex = Random(s, pms)

    x   = FPS(pex.x, 10)
    ref = np.array([[0.8965466 , 0.36756187],
                    [0.0202184 , 0.83261985],
                    [0.19999652, 0.01852179],
                    [0.7142413 , 0.99884701],
                    [0.45615033, 0.56843395],
                    [0.05802916, 0.43441663],
                    [0.7044144 , 0.03183893],
                    [0.34535168, 0.92808129],
                    [0.97861834, 0.79915856],
                    [0.46631077, 0.24442559]])

    assert np.allclose(x, ref)
