import sys

from sparkle.tst.runner import runner


###############################################
def test_pbo():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo.json",
           6.142433734222222e-04, 7.047430000000002e-06)

###############################################
def test_pbo_default():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo_default.json",
           6.142433734222222e-04, 7.047430000000002e-06)

###############################################
def test_pbo_mixed():

    # Add environment to PATH
    sys.path.append("sparkle/env/mixed")

    # Run test
    runner("sparkle/tst/agent/json/mixed_pbo.json",
           -3.689581619047619e+00,-5.999583333333334e+00)
