import sys

from sparkle.tst.runner import runner


###############################################
def test_cem():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_cem.json",
           1.431102162133334e-03, 6.048401466666668e-05)
