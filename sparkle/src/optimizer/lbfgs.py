import torch.optim as toptim


###############################################
### LBFGS optimizer class
class LBFGS():
    def __init__(self, model, pms):

        self.history_size = 20
        self.max_iter     = 5
        self.line_search  = "strong_wolfe"
        self.lr           = 1.0

        if (hasattr(pms, "history_size")): self.history_size = pms.history_size
        if (hasattr(pms, "max_iter")):     self.max_iter     = pms.max_iter
        if (hasattr(pms, "line_search")):  self.line_search  = pms.line_search
        if (hasattr(pms, "lr")):           self.lr           = pms.lr

        self.model_ = model
        self.opt_   = toptim.LBFGS(model.params(),
                                   history_size=self.history_size,
                                   max_iter=self.max_iter,
                                   line_search_fn=self.line_search,
                                   lr=self.lr)

    def zero_grad(self):

        self.opt_.zero_grad()

    def step(self, closure):

        self.opt_.step(closure)
