import os
import sys

from sparkle.tst.runner import runner


###############################################
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           4.429385405000000e-01, 1.790542405000000e-01)

    filename = "kriging.dat"
    os.remove(filename)
