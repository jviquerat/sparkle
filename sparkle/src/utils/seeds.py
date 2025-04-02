import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """
    Sets the random seeds for NumPy and PyTorch.

    Args:
        seed: The seed value to use.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
