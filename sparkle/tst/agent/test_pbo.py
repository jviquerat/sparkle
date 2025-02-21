# Generic imports
import pytest

# Custom imports
from sparkle.tst.tst    import *
from sparkle.tst.runner import *

###############################################
### Test pbo
def test_pbo():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo.json",
           "sparkle/tst/agent/data/parabola_pbo.dat",
           1.2965548977777777e-07, 3.329470133333333e-09)

###############################################
### Test pbo with default parameters
def test_pbo_default():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pbo_default.json",
           "sparkle/tst/agent/data/parabola_pbo.dat",
           1.2965548977777777e-07, 3.329470133333333e-09)
