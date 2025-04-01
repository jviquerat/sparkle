# Generic imports
import torch

# Custom imports
from sparkle.src.network.branched_mlp import BranchedMLP

###############################################
### Test branched_mlp
def test_branched_mlp():

    # branched mlp with input of size 3, output of size 1,
    # 2 layers of trunk, and 2 output branches with 2 layers in each
    net = BranchedMLP(inp_dim = 3,
                       out_dim = 1,
                       arch    = [[8,8],[[4],[2]]],
                       acts    = [["relu"],[["relu","linear"],["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert y.shape == (2,1,1)

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (2,4,1)

    # branched mlp with input of size 3, output of size 2,
    # 2 layers of trunk, and 2 output branches with 2 layers in each
    net = BranchedMLP(inp_dim = 3,
                       out_dim = 2,
                       arch    = [[8,8],[[4],[2]]],
                       acts    = [["relu"],[["relu","linear"],["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert y.shape == (2,1,2)

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (2,4,2)

    # branched mlp with input of size 3, output of size 2,
    # 2 layers of trunk, and 1 output branche with 1 layer
    net = BranchedMLP(inp_dim = 3,
                       out_dim = 2,
                       arch    = [[8,8],[[2]]],
                       acts    = [["relu"],[["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert y.shape == (1,1,2)

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (1,4,2)

    # branched mlp with specified activations for each layer of trunk
    net = BranchedMLP(inp_dim = 3,
                       out_dim = 2,
                       arch    = [[8,8],[[4],[2]]],
                       acts    = [["relu","tanh"],[["relu","linear"],["relu","linear"]]])

    # input with size 3 and batch_size 1
    x = torch.zeros(1,3)
    y = net.forward(x)
    assert y.shape == (2,1,2)

    # input with size 3 and batch_size 4
    x = torch.zeros(4,3)
    y = net.forward(x)
    assert y.shape == (2,4,2)
