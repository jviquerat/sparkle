import torch
import torch.nn as tnn
from torch import Tensor
torch.set_default_dtype(torch.double)

class LipschitzLinear(tnn.Module):
    """
    Linear layer with Lipschitz constraint.

    This class implements a linear layer with a spectral normalization
    constraint to enforce a Lipschitz constant.
    """
    def __init__(self, dim_in: int, dim_out: int, lip_constant: float=1.0) -> None:
        """
        Initializes the LipschitzLinear layer.

        Args:
            dim_in: The input dimension.
            dim_out: The output dimension.
            lip_constant: The Lipschitz constant to enforce.
        """
        super().__init__()

        # Lipschitz constant
        self.lip_constant = tnn.Parameter(torch.tensor(lip_constant))

        # Normalize weights to 1
        self.linear = tnn.Linear(dim_in, dim_out)
        self.linear = tnn.utils.spectral_norm(self.linear)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """

        scaled_weight = self.lip_constant*self.linear.weight
        return tnn.functional.linear(x, scaled_weight, self.linear.bias)
