import copy
from typing import List

import torch
import torch.nn as tnn

from sparkle.src.network.base import BaseNetwork
from sparkle.src.network.torch_dicts import add_mlp_layer
from sparkle.src.utils.error import error
from sparkle.src.utils.prints import new_line, spacer


class MLP(BaseNetwork):
    """
    Multi-Layer Perceptron (MLP) neural network.

    This class implements a standard Multi-Layer Perceptron, a feedforward
    neural network with one or more hidden layers.
    """
    def __init__(self,
                 inp_dim: int,
                 out_dim: int,
                 arch: List[int],
                 acts: List[str],
                 name: str="default") -> None:
        """
        Initializes the MLP.

        Args:
            inp_dim: The input dimension.
            out_dim: The output dimension.
            arch: A list of integers representing the number of units in each hidden layer.
            acts: A list of strings representing the activation function for each layer.
            name: An optional name for the network.
        """
        super().__init__()

        # I/O dimensions
        self.inp_dim_ = inp_dim
        self.out_dim_ = out_dim
        self.name_    = name

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

        self.net_ = tnn.ModuleList()

        # Add layers
        for k in range(0,len(self.arch_)-1):
            self.nf_ += add_mlp_layer(self.net_,
                                      self.arch_[k],
                                      self.arch_[k+1],
                                      self.acts_[k])

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

    def info(self) -> None:
        """
        Prints information about the network architecture.
        """

        new_line()
        spacer("MLP "+str(self.name_))
        spacer("Input layer, size "+str(self.inp_dim_))

        n = 0
        for k in range(0,len(self.arch_)-1):
            spacer("Layer "+str(n)+", size "+str(self.arch_[k+1])+", activation "+str(self.acts_[k]))
            n += 1
