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
           5.573511873000001e+00, 3.491620000000000e-02)

###############################################
### Test ego with loaded model
def test_ego_load_model():

    # Add environment to PATH
    sys.path.append("sparkle/env/sinebump")

    # Run test
    runner("sparkle/tst/agent/json/sinebump_ego_load.json",
           "sparkle/tst/agent/data/sinebump_ego_load.dat",
           5.397090666666666e-01, -1.772969333333333e+00)
