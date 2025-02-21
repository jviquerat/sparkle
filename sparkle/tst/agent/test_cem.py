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
           0.0005887422624000001, 5.550205866666667e-06)
