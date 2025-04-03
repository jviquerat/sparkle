import sys

from sparkle.tst.runner import runner


###############################################
### Test ego
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           1.286324780000000e+01, 9.951025999999998e-01)
