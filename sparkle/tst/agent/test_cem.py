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
    runner("sparkle/tst/agent/json/parabola_cem.json",
           0.0005887422624000001, 5.550205866666667e-06)
