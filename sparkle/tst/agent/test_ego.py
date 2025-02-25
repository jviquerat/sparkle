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
           2.253599800000000e-01, 1.480227800000000e-01)

###############################################
### Test ego with loaded model
def test_ego_load_model():

    # Add environment to PATH
    sys.path.append("sparkle/env/sinebump")

    # Run test
    runner("sparkle/tst/agent/json/sinebump_ego_load.json",
           "sparkle/tst/agent/data/sinebump_ego_load.dat",
           2.525011666666666e+00, -1.777113666666667e+00)
