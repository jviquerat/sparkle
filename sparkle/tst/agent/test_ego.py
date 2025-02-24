# Generic imports
import pytest

# Custom imports
from sparkle.tst.tst    import *
from sparkle.tst.runner import *

###############################################
### Test ego
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           "sparkle/tst/agent/data/parabola_ego.dat",
           5.229096616160000e+00, 7.654482999999999e-05)

###############################################
### Test ego with loaded model
def test_ego_load_model():

    # Add environment to PATH
    sys.path.append("sparkle/env/sinebump")

    # Run test
    runner("sparkle/tst/agent/json/sinebump_ego_load.json",
           "sparkle/tst/agent/data/sinebump_ego_load.dat",
           2.949909333333333e+00, -1.783417666666667e+00)
