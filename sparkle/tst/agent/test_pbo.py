import sys

from sparkle.tst.runner import runner


###############################################
def test_pbo():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo.json",
           6.065312728888891e-07, 1.248481933333334e-08)

###############################################
def test_pbo_default():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo_default.json",
           6.065312728888891e-07, 1.248481933333334e-08)

###############################################
def test_pbo_mixed():

    # Add environment to PATH
    sys.path.append("sparkle/env/mixed")

    # Run test
    runner("sparkle/tst/agent/json/mixed_pbo.json",
           -1.454045180952381e+00, -4.666466209523810e+00)
