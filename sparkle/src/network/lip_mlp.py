# Generic imports
import copy
import torch
import torch.nn as tnn
import torch.optim as toptim

# Custom imports
from sparkle.src.network.base        import base
from sparkle.src.network.torch_dicts import add_mlp_layer, add_lip_layer
from sparkle.src.utils.prints        import spacer, new_line
from sparkle.src.utils.error         import error

###############################################
### Lipschitz network
class lip_mlp(base):
    def __init__(self, inp_dim, out_dim, arch, acts, lip_constant=1.0, name="default"):
        super().__init__()

        # I/O dimensions
        self.inp_dim_      = inp_dim
        self.out_dim_      = out_dim
        self.lip_constant_ = lip_constant
        self.name_         = name

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
            self.nf_ += add_lip_layer(self.net_,
                                      self.arch_[k],
                                      self.arch_[k+1],
                                      self.acts_[k],
                                      self.lip_constant_)

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

        new_line()
        spacer("Lipschitz MLP "+str(self.name_))
        spacer("Lipschitz constant="+str(self.lip_constant_))
        spacer("Input layer, size "+str(self.inp_dim_))

        n = 0
        for k in range(0,len(self.arch_)-1):
            spacer("Layer "+str(n)+", size "+str(self.arch_[k+1])+", activation "+str(self.acts_[k]))
            n += 1
