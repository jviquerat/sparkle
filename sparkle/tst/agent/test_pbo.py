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
    runner("sparkle/tst/agent/parabola_pbo.json", "pbo",
           1.5180286994666663e-08, 1.5375755919999999e-09)
