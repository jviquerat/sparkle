# Generic imports
import math
import pytest

# Custom imports
from sparkle.tst.tst           import *
from sparkle.src.utils.compare import *

###############################################
### Test comparison
def test_compare():

    # Comparing true
    r1 = 1.0000000001
    r2 = 1.0
    re = 1.0e-9
    smaller = compare(r1,r2,re)
    assert(smaller == True)

    # Comparing false
    r1 = 1.0000000001
    r2 = 1.0
    re = 1.0e-11
    smaller = compare(r1,r2,re)
    assert(smaller == False)

    # Case that compares to true for weak and to false for strong
    r1 = 1.0e-10
    r2 = 1.1e-10
    re = 0.099
    smaller = compare(r1,r2,re)
    assert(smaller == False)

    # Test below min
    r1 = 0.1*sys.float_info.min
    r2 = 1.0
    re = sys.float_info.min
    smaller = compare(r1,r2,re)
    assert(smaller == True)
