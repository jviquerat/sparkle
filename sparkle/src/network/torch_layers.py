# Generic imports
import torch.nn as tnn
from torch import Tensor

###############################################
# A linear lipschitz layer
class LipschitzLinear(tnn.Module):
    def __init__(self, dim_in: int, dim_out: int, lip_constant: float=1.0) -> None:
        super().__init__()

        # Lipschitz constant
        self.lip_constant = lip_constant

        # Normalize weights to 1
        self.linear = tnn.Linear(dim_in, dim_out)
        self.linear = tnn.utils.spectral_norm(self.linear)

    # We scale the normalized weights by the lipschitz constant
    def forward(self, x: Tensor) -> Tensor:

        scaled_weight = self.lip_constant*self.linear.weight
        return tnn.functional.linear(x, scaled_weight, self.linear.bias)
