# Generic imports
import pytest

# Custom imports
from sparkle.tst.tst    import *
from sparkle.tst.runner import *

###############################################
### Test pbo
def test_pbo():

    # Add environment to PATH
    sys.path.append("sparkle/env/rosenbrock")

    # Run test
    runner("sparkle/tst/agent/rosenbrock_pbo.json",
           "pbo", 1.0e-5, 1.0e-7)
