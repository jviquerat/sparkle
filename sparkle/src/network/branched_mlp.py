import copy
import torch
import torch.nn as tnn
from typing import List, Union

from sparkle.src.network.base        import BaseNetwork
from sparkle.src.network.torch_dicts import add_mlp_layer
from sparkle.src.utils.prints        import spacer, new_line
from sparkle.src.utils.error         import error

###############################################
### Branched MLP class
class BranchedMLP(BaseNetwork):
    def __init__(self,
                 inp_dim: int,
                 out_dim: int,
                 arch: List[List[Union[int, List[int]]]],
                 acts: List[List[Union[str, List[str]]]],
                 name: str="default") -> None:
        super().__init__()

        # I/O dimensions
        self.inp_dim_ = inp_dim
        self.out_dim_ = out_dim
        self.name_    = name

        # Build architecture
        self.trunk_arch_ = arch[0]
        self.heads_arch_ = arch[1]
        self.n_heads_    = len(self.heads_arch_)
        self.nf_trunk_   = 0
        self.nf_heads_   = [0]*self.n_heads_
        self.trunk_arch_ = [inp_dim] + self.trunk_arch_
        for k in range(self.n_heads_):
            self.heads_arch_[k] = [self.trunk_arch_[-1]] + self.heads_arch_[k] + [out_dim]

        # Build activations
        self.trunk_acts_ = acts[0]
        self.heads_acts_ = acts[1]

        # Allow the use of a single activation for all trunk layers
        if ((len(self.trunk_acts_) == 1) and (len(self.trunk_arch_) > 1)):
            self.trunk_acts_ = [acts[0][0]]*len(self.trunk_arch_)

        # Check heads sizes
        if (self.n_heads_ != len(self.heads_acts_)):
            error("mlp", "__init__",
                  "The nb of heads does not match the nb of activations")

        # Check adequation between layers and activations
        for k in range(self.n_heads_):
            if (len(self.heads_acts_[k]) != len(self.heads_arch_[k])-1):
                error("branched_mlp", "__init__",
                      "Activations and architecture don't match")

        self.net_ = tnn.ModuleList()

        # Add trunk
        for k in range(0,len(self.trunk_arch_)-1):
            self.nf_trunk_ += add_mlp_layer(self.net_,
                                            self.trunk_arch_[k],
                                            self.trunk_arch_[k+1],
                                            self.trunk_acts_[k])

        # Add heads
        for h in range(self.n_heads_):
            for k in range(0,len(self.heads_arch_[h])-1):
                self.nf_heads_[h] += add_mlp_layer(self.net_,
                                                   self.heads_arch_[h][k],
                                                   self.heads_arch_[h][k+1],
                                                   self.heads_acts_[h][k])

        # Save model parameters in memory
        self.net_weights = copy.deepcopy(self.net_.state_dict())

    # Forward pass
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:

        # Initialize
        x          = torch.clone(x_in)
        batch_size = x.shape[0]
        out        = torch.zeros(self.n_heads_, batch_size, self.out_dim_)
        l          = 0

        # Forward pass in trunk
        for k in range(self.nf_trunk_):
            x  = self.net_[l](x)
            l += 1

        # Forward pass in heads
        for h in range(self.n_heads_):
            hx = torch.clone(x)
            for k in range(self.nf_heads_[h]):
                hx = self.net_[l](hx)
                l += 1
            out[h] = hx

        # Handle output shape
        return out

    # Reset
    def reset(self) -> None:

        self.net_.load_state_dict(self.net_weights)

    # Infos on network
    def info(self) -> None:

        new_line()
        spacer("Branched MLP "+str(self.name_))
        spacer("Input layer, size "+str(self.inp_dim_))
        spacer("Trunk:")

        n = 0
        for k in range(0,len(self.trunk_arch_)-1):
            spacer("Layer "+str(n)+", size "+str(self.trunk_arch_[k+1])+", activation "+str(self.trunk_acts_[k]))
            n += 1

        for h in range(self.n_heads_):
            spacer("Head "+str(h)+":")

            for k in range(0,len(self.heads_arch_[h])-1):
                spacer("Layer "+str(n)+", size "+str(self.heads_arch_[h][k+1])+", activation "+str(self.heads_acts_[h][k]))
                n += 1
