import sys

from sparkle.tst.runner import runner


###############################################
def test_sa():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_sa.json",
           2.168113746666667e-01, 8.732066666666666e-03)
