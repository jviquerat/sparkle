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
    runner("sparkle/tst/agent/parabola_ego.json",
           "ego", 100.0, 1.0e-3)

###############################################
### Test ego with loaded model
def test_ego_load_model():

    # Add environment to PATH
    sys.path.append("sparkle/env/sinebump")

    # Run test
    runner("sparkle/tst/agent/sinebump_ego_load.json",
           "ego", 100.0, 1.0e-3)
