# Generic imports
import copy
import torch
import torch.nn as tnn
import torch.optim as toptim

# Custom imports
from sparkle.src.network.base        import base
from sparkle.src.network.torch_dicts import add_mlp_layer
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
        self.arch_ = arch
        self.nf_   = 0

        # Fill output dimension in arch
        self.arch_[-1] = self.out_dim_

        # Build activations
        self.acts_ = acts

        # Allow the use of a single activation for all layers
        if ((len(self.acts_) == 1) and (len(self.arch_) > 1)):
            self.acts_ = [acts[0]]*len(self.arch_)

        self.net_ = tnn.ModuleList()

        # Add layers
        self.nf_ += add_mlp_layer(self.net_,
                                  self.inp_dim_,
                                  self.arch_[0],
                                  self.acts_[0])

        for k in range(1,len(self.arch_)):
            self.nf_ += add_mlp_layer(self.net_,
                                      self.arch_[k-1],
                                      self.arch_[k],
                                      self.acts_[k])

        # Save model parameters in memory
        self.net_weights = copy.deepcopy(self.net_.state_dict())

    # Forward pass
    def forward(self, x_in):

        # Initialize
        x = torch.clone(x_in)
        l = 0

        # Forward pass
        for k in range(self.nf_):
            x  = self.net_[l](x)
            l += 1

        return x

    # Reset
    def reset(self):

        self.net_.load_state_dict(self.net_weights)

    # Infos on network
    def info(self):

        spacer()
        print("MLP")

        spacer()
        print("Input layer, size "+str(self.inp_dim_))

        n = 0
        for k in range(len(self.arch_)):
            spacer()
            print("Layer "+str(n)+", size "+str(self.arch_[k])+", activation "+str(self.acts_[k]))
            n += 1
