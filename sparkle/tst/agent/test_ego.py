import os
import sys

from sparkle.tst.runner import runner


###############################################
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           1.790231330000000e+00, 6.684993000000000e-02)

    filename = "kriging.dat"
    os.remove(filename)
