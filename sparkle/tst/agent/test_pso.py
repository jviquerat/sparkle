# Generic imports
import pytest

# Custom imports
from sparkle.tst.tst    import *
from sparkle.tst.runner import *

###############################################
### Test pso
def test_pso():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pso.json",
           "sparkle/tst/agent/data/parabola_pso.dat",
           0.0005579100443309332, 1.2262383600000001e-07)
