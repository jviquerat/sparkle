import sys

from sparkle.tst.runner import runner

###############################################
### Test pbo
def test_pbo():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo.json",
           1.296554897777778e-07, 6.301704800000000e-09)

###############################################
### Test pbo with default parameters
def test_pbo_default():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo_default.json",
           1.296554897777778e-07, 6.301704800000000e-09)
