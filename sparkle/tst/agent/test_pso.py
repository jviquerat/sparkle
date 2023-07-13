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
    runner("sparkle/tst/agent/parabola_pso.json",
           "pso", 2.0e-3, 1.0e-5)
