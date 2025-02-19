# Generic imports
import math
import torch
import torch.nn as tnn
import numpy    as np

# Custom imports
from sparkle.src.network.torch_layers import lipschitz_linear

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

# Helper function to append a layer to a ModuleList
def add_mlp_layer(net, dim_in, dim_out, activation):

    n = 0
    net.append(tnn.Linear(dim_in, dim_out))

    std = math.sqrt(1.0/dim_in)
    tnn.init.normal_(net[-1].weight, mean=0.0, std=std)

    n += 1

    if (activation != "linear"):
        net.append(torch_activations[activation])
        n += 1

    return n

# Helper function to append a layer to a ModuleList
def add_lip_layer(net, dim_in, dim_out, activation, lip_constant):

    n = 0
    net.append(lipschitz_linear(dim_in, dim_out, lip_constant=lip_constant))

    n += 1

    if (activation != "linear"):
        net.append(torch_activations[activation])
        n += 1

    return n
