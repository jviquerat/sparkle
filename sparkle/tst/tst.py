# Generic imports
import warnings

# Filter warning messages
warnings.filterwarnings('ignore',category=DeprecationWarning)

# Compare two floats with given accuracy
def compare(x, y, eps=1.0e-8):

    return math.abs(x-y) < eps
