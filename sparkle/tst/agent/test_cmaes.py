# Generic imports
import pytest

# Custom imports
from sparkle.tst.tst    import *
from sparkle.tst.runner import *

###############################################
### Test cmaes
def test_cmaes():

    # Add environment to PATH
    sys.path.append("sparkle/env/rosenbrock")

    # Run test
    runner("sparkle/tst/agent/rosenbrock_cmaes.json", "cmaes",
           5.070978784924771e-14, 5.000620456522168e-15)
