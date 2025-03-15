# Generic imports
import warnings
import torch
import filecmp

# Filter warning messages
warnings.filterwarnings('ignore',category=DeprecationWarning)

# Distance between two torch tensors
def tensor_distance(x, y):

    return torch.linalg.vector_norm(x-y)

# Compare files
def compare_files(f1, f2):

    return filecmp.cmp(f1, f2, shallow=False)
