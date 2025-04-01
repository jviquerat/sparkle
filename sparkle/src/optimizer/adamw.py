# Generic imports
import torch.optim as toptim

###############################################
### AdamW optimizer class
class AdamW():
    def __init__(self, model, pms):

        self.model_ = model
        self.opt_   = toptim.AdamW(self.model_.params(),
                                  lr=pms.lr)

    def zero_grad(self):

        self.opt_.zero_grad()

    def step(self):

        self.opt_.step()
