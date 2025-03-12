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
           1.263791000000000e-01, 3.767919000000000e-02)

###############################################
### Test ego with loaded model
def test_ego_load_model():

    # Add environment to PATH
    sys.path.append("sparkle/env/sinebump")

    # Run test
    runner("sparkle/tst/agent/json/sinebump_ego_load.json",
           1.018526033333333e+00, -1.766342000000000e+00)
