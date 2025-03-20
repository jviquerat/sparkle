# Generic imports
import sys
import pytest

# Custom imports
from sparkle.tst.runner import runner

###############################################
### Test ego
def test_ego():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_ego.json",
           1.286324780000000e+01, 9.951025999999998e-01)

###############################################
### Test ego with loaded model
def test_ego_load_model():

    # Add environment to PATH
    sys.path.append("sparkle/env/sinebump")

    # Run test
    runner("sparkle/tst/agent/json/sinebump_ego_load.json",
           2.754777996000001e+00, -1.756220000000000e+00)
