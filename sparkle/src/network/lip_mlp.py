import copy
from typing import List

import torch
import torch.nn as tnn

from sparkle.src.network.base import BaseNetwork
from sparkle.src.network.torch_dicts import add_lip_layer
from sparkle.src.utils.error import error
from sparkle.src.utils.prints import new_line, spacer


class LipMLP(BaseNetwork):
    """
    Lipschitz Multi-Layer Perceptron (LipMLP) neural network.

    This class implements a Multi-Layer Perceptron with Lipschitz constraints
    on the linear layers, ensuring a bounded Lipschitz constant for the network.
    """
    def __init__(self,
                 inp_dim: int,
                 out_dim: int,
                 arch: List[int],
                 acts: List[str],
                 lip_constants: List[float]=[1.0],
                 name: str="default") -> None:
        """
        Initializes the LipMLP.

        Args:
            inp_dim: The input dimension.
            out_dim: The output dimension.
            arch: A list of integers representing the number of units in each hidden layer.
            acts: A list of strings representing the activation function for each layer.
            lip_constants: The Lipschitz constant for the linear layers.
            name: An optional name for the network.
        """
        super().__init__()

        # I/O dimensions
        self.inp_dim_   = inp_dim
        self.out_dim_   = out_dim
        self.lip_const_ = lip_constants
        self.name_      = name

        # Build architecture
        self.arch_ = arch
        self.nf_   = 0
        self.arch_ = [inp_dim] + self.arch_ + [out_dim]
        self.acts_ = acts

        # Allow the use of a single activation for all layers
        if ((len(self.acts_) == 1) and (len(self.arch_) > 1)):
            self.acts_ = [acts[0]]*(len(self.arch_)-1)

        # Check adequation between layers and activations
        if (len(self.acts_) != len(self.arch_)-1):
            error("mlp", "__init__",
                  "Activations and architecture don't match")

        # Check lipschitz constant list
        if (len(self.lip_const_) == 1):
            self.lip_const_ = [lip_constants[0]]*(len(self.arch_)-1)

        self.net_ = tnn.ModuleList()

        # Add layers
        for k in range(0,len(self.arch_)-1):
            self.nf_ += add_lip_layer(self.net_,
                                      self.arch_[k],
                                      self.arch_[k+1],
                                      self.acts_[k],
                                      self.lip_const_[k])

        # Save model parameters in memory
        self.net_weights = copy.deepcopy(self.net_.state_dict())

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x_in: The input tensor.

        Returns:
            The output tensor.
        """

        # Initialize
        x = torch.clone(x_in)
        l = 0

        # Forward pass
        for k in range(self.nf_):
            x  = self.net_[l](x)
            l += 1

        return x

    def reset(self) -> None:
        """
        Resets the network to its initial state.
        """

        self.net_.load_state_dict(self.net_weights)

    # Return lipschitz constants vector
    def lip_consts(self):

        consts = []
        for k in self.net_.named_parameters():
            if ("lip_constant" in k[0]):
                consts.append(k[1].item())

        return consts

    def info(self) -> None:
        """
        Prints information about the network architecture.
        """

        new_line()
        spacer("Lipschitz MLP "+str(self.name_))
        spacer("Lipschitz constant="+str(self.lip_const_))
        spacer("Input layer, size "+str(self.inp_dim_))

        n = 0
        for k in range(0,len(self.arch_)-1):
            spacer("Layer "+str(n)+", size "+str(self.arch_[k+1])+", activation "+str(self.acts_[k]))
            n += 1
