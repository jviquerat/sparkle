import torch.optim as toptim
from types import SimpleNamespace

from sparkle.src.network.mlp import MLP

###############################################
### AdamW optimizer class
class AdamW():
    def __init__(self, model: MLP, pms: SimpleNamespace) -> None:

        self.model_ = model
        self.opt_   = toptim.AdamW(self.model_.params(),
                                  lr=pms.lr)

    def zero_grad(self) -> None:

        self.opt_.zero_grad()

    def step(self) -> None:

        self.opt_.step()
