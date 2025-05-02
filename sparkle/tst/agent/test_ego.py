import os
import sys

from sparkle.tst.runner import runner


###############################################
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           1.164042800000000e-01, 2.883067260000000e-02)

    filename = "kriging.dat"
    os.remove(filename)
