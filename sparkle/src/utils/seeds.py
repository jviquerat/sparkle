import torch
import numpy as np

# Set numpy and torch seeds
def set_seeds(seed: int) -> None:

    torch.manual_seed(seed)
    np.random.seed(seed)
