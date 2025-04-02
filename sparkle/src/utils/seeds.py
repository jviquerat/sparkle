import numpy as np
import torch


# Set numpy and torch seeds
def set_seeds(seed: int) -> None:

    torch.manual_seed(seed)
    np.random.seed(seed)
