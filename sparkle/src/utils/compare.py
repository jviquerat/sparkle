# Generic imports
import sys
import math

# Relative comparison of two floats with given accuracy
def compare(x, y, eps=1.0e-8):

    d = math.fabs(x-y);
    m = min(math.fabs(x), math.fabs(y));

    if (d < sys.float_info.min or m < sys.float_info.min): return True
    if (m > sys.float_info.epsilon):
        return (d/m < eps)
    else:
        return (d/math.sqrt(m) < abs(eps)*math.sqrt(m))
