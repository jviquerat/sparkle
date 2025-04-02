import torch

from sparkle.src.network.mlp import MLP

###############################################
### Test mlp
def test_mlp():

    # mlp with input of size 3, output of size 1
    net = MLP(inp_dim = 3,
              out_dim = 1,
              arch    = [8,8,8],
              acts    = ["relu"])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert y.shape == (1,1)

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (4,1)

    # mlp with input of size 3, output of size 2
    net = MLP(inp_dim = 3,
              out_dim = 2,
              arch    = [8,8,8],
              acts    = ["relu"])

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (4,2)

    # mlp with specified activations for each layer
    net = MLP(inp_dim = 3,
              out_dim = 2,
              arch    = [8,8,8],
              acts    = ["relu","relu","relu","linear"])

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (4,2)
