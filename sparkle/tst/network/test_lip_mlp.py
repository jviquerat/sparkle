# Generic imports
import torch

# Custom imports
from sparkle.src.utils.distances import tensor_distance
from sparkle.src.network.lip_mlp import LipMLP

###############################################
### Test lipschitz mlp
def test_lip_mlp():

    # lip mlp with input of size 3, output of size 1, lip_constant of 1
    net = LipMLP(inp_dim      = 3,
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
    assert dy <= dx

    # lip mlp with input of size 3, output of size 1, lip_constant of 0.1
    net = LipMLP(inp_dim      = 3,
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
    assert dy <= 0.01*dx

    # lip mlp with all specified activations
    net = LipMLP(inp_dim      = 3,
                  out_dim      = 1,
                  arch         = [8,8,8],
                  acts         = ["relu","relu","relu","linear"],
                  lip_constant = 0.01)

    # input with size 3 and batch_size 1
    x0 = 1.0*torch.ones(1,3)
    x1 = 100.0*torch.ones(1,3)
    y0 = net.forward(x0)
    y1 = net.forward(x1)
    dx = tensor_distance(x0, x1)
    dy = tensor_distance(y0, y1)
    assert dy <= 0.01*dx
