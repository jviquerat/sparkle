# Generic imports
import pytest
import torch
import numpy as np

# Custom imports
from sparkle.tst.tst                  import *
from sparkle.src.network.torch_dicts  import set_seeds
from sparkle.src.network.mlp          import mlp
from sparkle.src.network.branched_mlp import branched_mlp
from sparkle.src.network.lip_mlp      import lip_mlp

###############################################
### Test mlp
def test_mlp():

    # mlp with input of size 3, output of size 1
    net = mlp(inp_dim = 3,
              out_dim = 1,
              arch    = [8,8,8],
              acts    = ["relu"])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert(y.shape == (1,1))

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (4,1))

    # mlp with input of size 3, output of size 2
    net = mlp(inp_dim = 3,
              out_dim = 2,
              arch    = [8,8,8],
              acts    = ["relu"])

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (4,2))

    # mlp with specified activations for each layer
    net = mlp(inp_dim = 3,
              out_dim = 2,
              arch    = [8,8,8],
              acts    = ["relu","relu","relu"])

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (4,2))

###############################################
### Test branched_mlp
def test_branched_mlp():

    set_seeds(0)

    # branched mlp with input of size 3, output of size 1,
    # 2 layers of trunk, and 2 output branches with 2 layers in each
    net = branched_mlp(inp_dim = 3,
                       out_dim = 1,
                       arch    = [[8,8],[[4,4],[2,2]]],
                       acts    = [["relu"],[["relu","linear"],["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert(y.shape == (2,1,1))

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (2,4,1))

    # branched mlp with input of size 3, output of size 2,
    # 2 layers of trunk, and 2 output branches with 2 layers in each
    net = branched_mlp(inp_dim = 3,
                       out_dim = 2,
                       arch    = [[8,8],[[4,4],[2,2]]],
                       acts    = [["relu"],[["relu","linear"],["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert(y.shape == (2,1,2))

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (2,4,2))

    # branched mlp with input of size 3, output of size 2,
    # 2 layers of trunk, and 1 output branche with 1 layer
    net = branched_mlp(inp_dim = 3,
                       out_dim = 2,
                       arch    = [[8,8],[[2]]],
                       acts    = [["relu"],[["linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert(y.shape == (1,1,2))

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (1,4,2))

    # branched mlp with specified activations for each layer of trunk
    net = branched_mlp(inp_dim = 3,
                       out_dim = 2,
                       arch    = [[8,8],[[4,4],[2,2]]],
                       acts    = [["relu","tanh"],[["relu","linear"],["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert(y.shape == (2,1,2))

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert(y.shape == (2,4,2))

###############################################
### Test lipschitz mlp
def test_lip_mlp():

    # lip mlp with input of size 3, output of size 1, lip_constant of 1
    net = lip_mlp(inp_dim      = 3,
                  out_dim      = 1,
                  arch         = [8,8,8],
                  acts         = ["relu"],
                  lip_constant = 1.0)

    # input with size 3 and batch_size 1
    x0 = 1.0*torch.ones(1,3)
    x1 = 2.0*torch.ones(1,3)
    y0 = net.forward(x0)
    y1 = net.forward(x1)
    dx = tensor_distance(x0, x1)
    dy = tensor_distance(y0, y1)
    assert(dy <= dx)

    # lip mlp with input of size 3, output of size 1, lip_constant of 0.1
    net = lip_mlp(inp_dim      = 3,
                  out_dim      = 1,
                  arch         = [8,8,8],
                  acts         = ["relu"],
                  lip_constant = 0.01)

    # input with size 3 and batch_size 1
    x0 = 1.0*torch.ones(1,3)
    x1 = 100.0*torch.ones(1,3)
    y0 = net.forward(x0)
    y1 = net.forward(x1)
    dx = tensor_distance(x0, x1)
    dy = tensor_distance(y0, y1)
    assert(dy <= 0.01*dx)
