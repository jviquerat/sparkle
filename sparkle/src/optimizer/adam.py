# Generic imports
import torch.optim as toptim

###############################################
### Adam optimizer class
class adam():
    def __init__(self, model, pms):

        self.model_ = model
        self.opt_   = toptim.Adam(self.model_.params(),
                                  lr=pms.lr)

    def zero_grad(self):

        self.opt_.zero_grad()

    def step(self):

        self.opt_.step()
