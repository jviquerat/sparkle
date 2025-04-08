import sys

from sparkle.tst.runner import runner


###############################################
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           1.768947175700000e+00, 1.873832957000000e-01)
