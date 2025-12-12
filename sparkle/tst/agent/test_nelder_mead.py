import sys

from sparkle.tst.runner import runner


###############################################
def test_nelder_mead():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_nelder_mead.json",
           5.872935605333332e-25, 7.744030586666669e-26)
