import os
import sys

from sparkle.tst.runner import runner


###############################################
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           5.929662800000001e-01, 5.331920000000001e-02)

    filename = "kriging.dat"
    os.remove(filename)
