# Generic imports
import copy
import torch
import torch.nn as tnn
import torch.optim as toptim

# Custom imports
from sparkle.src.network.base        import base
from sparkle.src.network.torch_dicts import add_fc_layer
from sparkle.src.utils.prints        import spacer

###############################################
### MLP class
class mlp(base):
    def __init__(self, inp_dim, out_dim, arch, acts):
        super().__init__()

        # I/O dimensions
        self.inp_dim_ = inp_dim
        self.out_dim_ = out_dim

        # Build architecture
        self.trunk_arch_ = arch[0]
        self.heads_arch_ = arch[1]
        self.n_heads_    = len(self.heads_arch_)
        self.nf_trunk_   = 0
        self.nf_heads_   = [0]*self.n_heads_

        # Fill output dimension in heads arch
        for h in range(self.n_heads_):
            self.heads_arch_[h][-1] = self.out_dim_

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

        self.net_ = tnn.ModuleList()

        # Add trunk
        self.nf_trunk_ += add_fc_layer(self.net_,
                                       self.inp_dim_,
                                       self.trunk_arch_[0],
                                       self.trunk_acts_[0])

        for k in range(1,len(self.trunk_arch_)):
            self.nf_trunk_ += add_fc_layer(self.net_,
                                           self.trunk_arch_[k-1],
                                           self.trunk_arch_[k],
                                           self.trunk_acts_[k])

        # Add heads
        for h in range(self.n_heads_):
            self.nf_heads_[h] += add_fc_layer(self.net_,
                                              self.trunk_arch_[-1],
                                              self.heads_arch_[h][0],
                                              self.heads_acts_[h][0])

            for k in range(1,len(self.heads_arch_[h])):
                self.nf_heads_[h] += add_fc_layer(self.net_,
                                                  self.heads_arch_[h][k-1],
                                                  self.heads_arch_[h][k],
                                                  self.heads_acts_[h][k])

        # Save model parameters in memory
        self.net_weights = copy.deepcopy(self.net_.state_dict())

    # Forward pass
    def forward(self, x_in):

        # Initialize
        x   = torch.clone(x_in)
        out = []
        l   = 0

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
            out.append(hx)

        # Handle output shape
        if (len(out) == 1): return out[0]
        else: return out

    # Reset
    def reset(self):

        self.net_.load_state_dict(self.net_weights)

    # Infos on network
    def info(self):

        spacer()
        print("Input layer, size "+str(self.inp_dim_))

        spacer()
        print("Trunk:")

        n = 0
        for k in range(len(self.trunk_arch_)):
            spacer()
            print("Layer "+str(n)+", size "+str(self.trunk_arch_[k])+", activation "+str(self.trunk_acts_[k]))
            n += 1

        for h in range(self.n_heads_):
            spacer()
            print("Head "+str(h)+":")

            for k in range(len(self.heads_arch_[h])):
                spacer()
                print("Layer "+str(n)+", size "+str(self.heads_arch_[h][k])+", activation "+str(self.heads_acts_[h][k]))
                n += 1
