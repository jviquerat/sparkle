import sys

from sparkle.tst.runner import runner


###############################################
def test_pbo():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo.json",
           1.138438497333333e-03, 3.937409355555556e-05)

###############################################
def test_pbo_default():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo_default.json",
           1.138438497333333e-03, 3.937409355555556e-05)
