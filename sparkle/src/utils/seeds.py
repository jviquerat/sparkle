# Generic imports
import torch
import numpy as np

# Set numpy and torch seeds
def set_seeds(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
