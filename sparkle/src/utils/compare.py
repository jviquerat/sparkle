import sys
import math
from numpy import float64

# Relative comparison of two floats with given accuracy
def compare(x: float, y: float, eps: float=1.0e-8) -> bool:

    d = math.fabs(x-y);
    m = min(math.fabs(x), math.fabs(y));

    if (d < sys.float_info.min or m < sys.float_info.min): return True
    if (m > sys.float_info.epsilon):
        return (d/m < eps)
    else:
        return (d/math.sqrt(m) < abs(eps)*math.sqrt(m))
