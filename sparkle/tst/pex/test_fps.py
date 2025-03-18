# Generic imports
import numpy as np

# Custom imports
from sparkle.src.pex.fps     import fps
from sparkle.src.utils.seeds import set_seeds

###############################################
### Test furthest point sampling
def test_fps():

    set_seeds(0)

    x = [np.array([0.0, 0.0]),
         np.array([0.1, 0.7]),
         np.array([0.4, 0.2]),
         np.array([0.4, 0.9]),
         np.array([0.8, 0.1]),
         np.array([0.2, 0.2])]

    x_fps = fps(x, 4)
    ref = np.array([[0.8, 0.1],
                    [0.1, 0.7],
                    [0.0, 0.0],
                    [0.4, 0.2]])

    assert(np.allclose(x_fps, ref))
