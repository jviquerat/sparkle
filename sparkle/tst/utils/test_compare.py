import sys

from sparkle.src.utils.compare import compare


###############################################
def test_compare():

    # Comparing true
    r1 = 1.0000000001
    r2 = 1.0
    re = 1.0e-9
    assert compare(r1,r2,re)

    # Comparing false
    r1 = 1.0000000001
    r2 = 1.0
    re = 1.0e-11
    assert not compare(r1,r2,re)

    # Case that compares to true for weak and to false for strong
    r1 = 1.0e-10
    r2 = 1.1e-10
    re = 0.099
    assert not compare(r1,r2,re)

    # Test below min
    r1 = 0.1*sys.float_info.min
    r2 = 1.0
    re = sys.float_info.min
    assert compare(r1,r2,re)
