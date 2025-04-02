import math
import sys

from numpy import float64


# Relative comparison of two floats with given accuracy
def compare(x: float, y: float, eps: float=1.0e-8) -> bool:
    """
    Compares two floating-point numbers for approximate equality.

    Args:
        x: The first floating-point number.
        y: The second floating-point number.
        eps: The relative tolerance for the comparison.

    Returns:
        True if the numbers are approximately equal, False otherwise.
    """

    d = math.fabs(x-y);
    m = min(math.fabs(x), math.fabs(y));

    if (d < sys.float_info.min or m < sys.float_info.min): return True
    if (m > sys.float_info.epsilon):
        return (d/m < eps)
    else:
        return (d/math.sqrt(m) < abs(eps)*math.sqrt(m))
