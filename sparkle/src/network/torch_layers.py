# Generic imports
import torch.nn as tnn

###############################################
# A linear lipschitz layer
class lipschitz_linear(tnn.Module):
    def __init__(self, dim_in, dim_out, lip_constant=1.0):
        super().__init__()

        # Lipschitz constant
        self.lip_constant = lip_constant

        # Normalize weights to 1
        self.linear = tnn.Linear(dim_in, dim_out)
        self.linear = tnn.utils.spectral_norm(self.linear)

    # We scale the normalized weights by the lipschitz constant
    def forward(self, x):

        scaled_weight = self.lip_constant*self.linear.weight
        return tnn.functional.linear(x, scaled_weight, self.linear.bias)
