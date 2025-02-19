# Generic imports
import warnings
import torch

# Filter warning messages
warnings.filterwarnings('ignore',category=DeprecationWarning)

# Compare two floats with given accuracy
def compare(x, y, eps=1.0e-8):

    return math.abs(x-y) < eps

# Distance between two torch tensors
def tensor_distance(x, y):

    return torch.linalg.vector_norm(x-y)
