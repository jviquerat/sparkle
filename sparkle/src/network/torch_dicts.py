import math

import torch
import torch.nn as tnn
from torch.nn.modules.container import ModuleList

from sparkle.src.network.torch_layers import LipschitzLinear

torch.set_default_dtype(torch.float64)

torch_activations = {
    'relu': tnn.ReLU(),
    'prelu': tnn.PReLU(),
    'relu6': tnn.ReLU6(),
    'lrelu': tnn.LeakyReLU(negative_slope=0.2),
    'selu': tnn.SELU(),
    'elu': tnn.ELU(),
    'mish': tnn.Mish(),
    'sigmoid': tnn.Sigmoid(),
    'tanh': tnn.Tanh(),
    'hardtanh': tnn.Hardtanh(),
    'softmax': tnn.Softmax(),
    'logsoftmax': tnn.LogSoftmax(),
    'linear': None
}


def add_mlp_layer(net: ModuleList,
                  dim_in: int,
                  dim_out: int,
                  activation: str) -> int:
    """
    Adds a standard linear layer with optional activation to a ModuleList.

    Args:
        net: The ModuleList to which to add the layer.
        dim_in: The input dimension.
        dim_out: The output dimension.
        activation: The activation function to use.

    Returns:
        The number of layers added.
    """

    n = 0
    net.append(tnn.Linear(dim_in, dim_out))

    std = math.sqrt(1.0/dim_in)
    tnn.init.normal_(net[-1].weight, mean=0.0, std=std)

    n += 1

    if (activation != "linear"):
        net.append(torch_activations[activation])
        n += 1

    return n


def add_lip_layer(net: ModuleList,
                  dim_in: int,
                  dim_out: int,
                  activation: str,
                  lip_constant: float) -> int:
    """
    Adds a Lipschitz-constrained linear layer with optional activation to a ModuleList.

    Args:
        net: The ModuleList to which to add the layer.
        dim_in: The input dimension.
        dim_out: The output dimension.
        activation: The activation function to use.
        lip_constant: The Lipschitz constant to enforce.

    Returns:
        The number of layers added.
    """

    n = 0
    net.append(LipschitzLinear(dim_in, dim_out, lip_constant=lip_constant))

    n += 1

    if (activation != "linear"):
        net.append(torch_activations[activation])
        n += 1

    return n
