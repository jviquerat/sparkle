# Generic imports
import pytest

# Custom imports
from sparkle.tst.tst    import *
from sparkle.tst.runner import *

###############################################
### Test cem
def test_cem():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_cem.json",
           "sparkle/tst/agent/data/parabola_cem.dat",
           4.817921056000001e-04, 5.619510400000001e-06)
