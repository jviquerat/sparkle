# Generic imports
import sys
import pytest

# Custom imports
from sparkle.tst.runner import runner

###############################################
### Test pso
def test_pso():

    # Add environment to PATH
    sys.path.append("sparkle/env/parabola")

    # Run test
    runner("sparkle/tst/agent/json/parabola_pso.json",
           5.579100443309332e-04, 1.418414712000001e-07)
