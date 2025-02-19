# Generic imports
import warnings
import torch
import numpy as np

# Filter warning messages
warnings.filterwarnings('ignore',category=DeprecationWarning)

# Set numpy and torch seeds
def set_seeds(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)

# Distance between two torch tensors
def tensor_distance(x, y):

    return torch.linalg.vector_norm(x-y)
